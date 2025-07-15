import tkinter as tk
from tkinter import filedialog, Toplevel

from ttkbootstrap import Style

import torch
import torchvision

from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import DataLoader

from torchvision import models, datasets
from torchvision.transforms import v2
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC    

import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages

import sklearn

from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

import numpy as np
import zipfile
import pathlib
import requests
import threading
import math

class BaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep Learning Trainer - ViT / CNN MRI Research")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.stop_training_var = False

        self.model = tk.StringVar(value="Select Model")
        self.epochs = tk.IntVar(value=5)
        self.batch_size = tk.IntVar(value=16)
        self.lr = tk.DoubleVar(value=0.001)
        self.dataset = tk.StringVar(value="Select Dataset")
        self.data_augmentation = tk.BooleanVar(value=False)

        self.dataset_options = ["Official - Kaggle Brain MRI"]
        self.model_options = ["EfficientNetB0", "EfficientNetB5", "EfficientNetB7","ViT-B16", "ViT-B32", "SwinV2Base", "ConvNeXtBase"]

        self.MODEL_HYPERPARAMETERS = {
            "SwinV2Base": {
                "lr": 1e-4,
                "batch_size": 16,
                "weight_decay": 0.01,
                "epochs": 60,
                "freeze_epochs": 5,
                "warmup_epochs": 6
            },
            "ConvNeXtBase": {
                "lr": 1e-4,
                "batch_size": 16,
                "weight_decay": 0.01,
                "epochs": 60,
                "freeze_epochs": 5,
                "warmup_epochs": 6
            },
            "ViT-B16": {
                "lr": 1e-4,
                "batch_size": 32,
                "weight_decay": 0.01,
                "epochs": 50,
                "freeze_epochs": 5,
                "warmup_epochs": 6
            },
            "ViT-B32": {
                "lr": 1e-4,
                "batch_size": 32,
                "weight_decay": 0.01,
                "epochs": 50,
                "freeze_epochs": 5,
                "warmup_epochs": 6
            },
            "EfficientNetB0": {
                "lr": 5e-4,
                "batch_size": 32,
                "weight_decay": 0.01,
                "epochs": 30,
                "freeze_epochs": 5,
                "warmup_epochs": 3
            },
            "EfficientNetB5": {
                "lr": 3e-4,
                "batch_size": 16,
                "weight_decay": 0.01,
                "epochs": 50,
                "freeze_epochs": 5,
                "warmup_epochs": 5
            },
            "EfficientNetB7": {
                "lr": 1e-4,
                "batch_size": 8,
                "weight_decay": 0.01,
                "epochs": 60,
                "freeze_epochs": 5,
                "warmup_epochs": 6
            }
        }

        self.build_gui()

    def on_close(self):
        self.stop_training_var = True
        self.root.destroy()

    def build_gui(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        tk.Label(frame, text="Model:").grid(row=0, column=0, sticky="w")
        tk.OptionMenu(frame, self.model, *self.model_options, command=self.handle_model_selection).grid(row=0, column=1, sticky="ew")

        tk.Label(frame, text="Epochs:").grid(row=1, column=0, sticky="w")
        tk.Entry(frame, textvariable=self.epochs).grid(row=1, column=1, sticky="ew")

        tk.Label(frame, text="Batch Size:").grid(row=2, column=0, sticky="w")
        tk.Entry(frame, textvariable=self.batch_size).grid(row=2, column=1, sticky="ew")

        tk.Label(frame, text="Learning Rate:").grid(row=3, column=0, sticky="w")
        tk.Entry(frame, textvariable=self.lr).grid(row=3, column=1, sticky="ew")

        tk.Label(frame, text="Dataset:").grid(row=4, column=0, sticky="w")
        tk.OptionMenu(frame, self.dataset, *self.dataset_options, command=self.handle_dataset_selection).grid(row=4, column=1, sticky="ew")

        tk.Label(frame, text="Data Augmentation:").grid(row=5, column=0, sticky="w")
        tk.Checkbutton(frame, variable=self.data_augmentation, text="Enable Augmentation",
                       command=self.log_data_augmentation).grid(row=5, column=1, sticky="w")
        
        
        self.output = tk.Text(self.root, height=10, width=70)
        self.output.pack(pady=5)

        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, pady=10)

        self.train_button = tk.Button(button_frame, text="Start Training", command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=10)

        self.end_train_button = tk.Button(button_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.end_train_button.pack(side=tk.LEFT, padx=10)

    def handle_dataset_selection(self, value):
        self.log(f"[INFO] Selected Dataset: {self.dataset.get()}")
    
    def handle_model_selection(self, value):
        self.lr.set(self.MODEL_HYPERPARAMETERS[value]["lr"])
        self.batch_size.set(self.MODEL_HYPERPARAMETERS[value]["batch_size"])
        self.epochs.set(self.MODEL_HYPERPARAMETERS[value]["epochs"])

        self.log(f"[INFO] Selected Model: {value}, Loaded Optimal Hyperparameters")

    def log_data_augmentation(self):
        if self.data_augmentation.get():
            self.log("[INFO] Data Augmentation Enabled")
        else:
            self.log("[INFO] Data Augmentation Disabled")


    def log(self, message):
        self.output.config(state=tk.NORMAL)
        self.output.insert(tk.END, message + "\n")
        self.output.see(tk.END)
        self.output.config(state=tk.DISABLED)
    
    def download_dataset(self, dataset: str):
        if dataset == 'Official - Kaggle Brain MRI':
            DATA_PATH = pathlib.Path("data")

            if not DATA_PATH.is_dir():
                self.log(message="[INFO] Downloading Dataset")
                self.root.update_idletasks()

                DATA_PATH.mkdir(parents=True, exist_ok=True)

                response = requests.get("https://github.com/zaina-ml/Brain-Tumor-MRI-Dataset/raw/refs/heads/main/archive.zip")

                with open(DATA_PATH / "archive.zip", "wb") as f:
                    f.write(response.content)

                with zipfile.ZipFile(DATA_PATH / "archive.zip", "r") as zipref:
                    zipref.extractall(DATA_PATH)
            
                self.log(message="[INFO] Finished Dataset Download")
                self.root.update_idletasks()
            else:
                self.log(message="[INFO] Dataset Already Downloaded")
                self.root.update_idletasks()

            train_dir = DATA_PATH / "Training"
            test_dir = DATA_PATH / "Testing"

            return train_dir, test_dir

    def load_transforms(self, augmentation: bool):
        self.log(message="[INFO] Loading Transforms")
        self.root.update_idletasks()

        if augmentation:
            train_transform = v2.Compose([
                v2.ToImage(),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply([v2.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02))], p=0.3),
                v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.2),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5], std=[0.5])
            ]) 

            test_transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5], std=[0.5])
            ])
            
            self.log(message="[INFO] Transforms Loaded")
            self.root.update_idletasks()
            
            return train_transform, test_transform
            
        elif not augmentation:
            train_transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(size=(224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5], std=[0.5])
            ])

            test_transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(size=(224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5], std=[0.5])
            ])
            
            self.log(message="[INFO] Transforms Loaded")
            self.root.update_idletasks()

            return train_transform, test_transform

    def create_model(self, architecture: str, device: str):
        self.log(message="[INFO] Loading Model")
        self.root.update_idletasks()

        if architecture == 'EfficientNetB0':
            weights = models.EfficientNet_B0_Weights.DEFAULT

            model = models.efficientnet_b0(weights=weights)
            model = model.to(device)

            for param in model.features.parameters():
                param.requires_grad = False

            in_features = model.classifier[1].in_features

            model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 4)
            ).to(device)

            self.log(message="[INFO] Model Loaded")
            self.root.update_idletasks()

            return model
        
        elif architecture == 'EfficientNetB5':
            weights = models.EfficientNet_B5_Weights.DEFAULT

            model = models.efficientnet_b5(weights=weights)
            model = model.to(device)

            for param in model.features.parameters():
                param.requires_grad = False

            in_features = model.classifier[1].in_features

            model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 4)
            ).to(device)

            self.log(message="[INFO] Model Loaded")
            self.root.update_idletasks()

            return model
        elif architecture == 'EfficientNetB7':
            weights = models.EfficientNet_B7_Weights.DEFAULT

            model = models.efficientnet_b7(weights=weights)
            model = model.to(device)

            for param in model.features.parameters():
                param.requires_grad = False

            in_features = model.classifier[1].in_features

            model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 4)
            ).to(device)

            self.log(message="[INFO] Model Loaded")
            self.root.update_idletasks()

            return model
        
        elif architecture == 'ViT-B16':
            weights = models.ViT_B_16_Weights.DEFAULT
            model = models.vit_b_16(weights=weights)

            in_features = model.heads.head.in_features
            
            model.heads.head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 4)
            ).to(device)

            for param in model.parameters():
                param.requires_grad = False

            for param in model.heads.head.parameters():
                param.requires_grad = True

            model = model.to(device)

            self.log(message="[INFO] Model Loaded")
            self.root.update_idletasks()

            return model
        
        elif architecture == 'ViT-B32':
            weights = models.ViT_B_32_Weights.DEFAULT
            model = models.vit_b_32(weights=weights)

            in_features = model.heads.head.in_features
            
            model.heads.head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 4)
            ).to(device)

            for param in model.parameters():
                param.requires_grad = False

            for param in model.heads.head.parameters():
                param.requires_grad = True

            model = model.to(device)

            self.log(message="[INFO] Model Loaded")
            self.root.update_idletasks()

            return model

        elif architecture == 'SwinV2Base':
            weights = models.Swin_V2_B_Weights.DEFAULT
            model = models.swin_v2_b(weights=weights)

            self.log(message="[INFO] Model Loaded")
            self.root.update_idletasks()
            
            in_features = model.head.in_features

            model.head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 4)
            ).to(device)

            for param in model.parameters():
                param.requires_grad = False

            for param in model.head.parameters():
                param.requires_grad = True

            return model
        elif architecture == 'ConvNeXtBase':
            weights = models.ConvNeXt_Base_Weights.DEFAULT
            model = models.convnext_base(weights=weights)
            print(model)

            self.log(message="[INFO] Model Loaded")
            self.root.update_idletasks()
            
            in_features = model.classifier[2].in_features

            model.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                nn.LayerNorm(in_features, eps=1e-06, elementwise_affine=True),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 4)
            ).to(device)

            for param in model.parameters():
                param.requires_grad = False

            for param in model.classifier.parameters():
                param.requires_grad = True

            return model

    def cosine_scheduler_with_warmup(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch) / float(max(1, warmup_epochs))
            else:
                progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

                return max(eta_min / self.lr.get(), cosine)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    
    def training_loop(self, 
                      epochs, 
                      model, 
                      loss_fn, 
                      optimizer, 
                      train_dataloader, 
                      test_dataloader, 
                      device,
                      test_dataset):
        
        self.epoch_list = []
        self.train_acc_list = []
        self.train_loss_list = []
        self.test_acc_list = []
        self.test_loss_list = []
        
        if device == 'cuda':
            scaler = GradScaler()

        scheduler = self.cosine_scheduler_with_warmup(optimizer=optimizer, warmup_epochs=self.MODEL_HYPERPARAMETERS[self.model.get()]["warmup_epochs"], total_epochs=epochs)
        
        self.log(message="[INFO] Beginning Training")
        self.training_log(message="[INFO] Beginning Training With Following Hyperparameters: \n")
        self.training_log(message=f"    Model: {self.model.get()}\n    Epochs: {self.epochs.get()}\n    Batch Size: {self.batch_size.get()}\n    Learning Rate: {self.lr.get()}\n    Dataset: {self.dataset.get()}\n    Data Augmentation: {self.data_augmentation.get()}\n")
        self.root.update_idletasks()

        if device == 'cpu':
            self.training_log(message="[WARNING] CUDA is not available. Train time may be extended")
            self.root.update_idletasks()
        
        for epoch in range(epochs):
            train_loss, test_loss = 0, 0
            train_acc, test_acc = 0, 0

            if epochs == self.MODEL_HYPERPARAMETERS[self.model.get()]["freeze_epochs"]:
                for param in model.parameters():
                    param.requires_grad = True

            for batch, (X, y) in enumerate(train_dataloader):
                if self.stop_training_var:
                    self.root.after(0, lambda: self.training_log("[INFO] Training Stopped"))
                    self.stop_training_var = False

                    return None
            
                print(batch)
                model.train()

                X, y = X.to(device), y.to(device)

                if device == 'cpu':
                    y_hat = model(X)
                    loss = loss_fn(y_hat, y)
        
                    optimizer.zero_grad(set_to_none=True)
        
                    loss.backward()
        
                    optimizer.step()
                elif device == 'cuda':
                    with autocast():
                        y_hat = model(X)
                        loss = loss_fn(y_hat, y)
        
                    optimizer.zero_grad(set_to_none=True)
        
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
    
                train_acc += ((torch.eq(y, y_hat.argmax(dim=1)).sum().item()) / len(y)) * 100
                train_loss += loss.item()
            scheduler.step()

            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)
            
            with torch.inference_mode():
                for batch, (X, y) in enumerate(test_dataloader):
                    X, y = X.to(device), y.to(device)

                    if self.stop_training_var:
                        self.root.after(0, lambda: self.training_log("[INFO] Training Stopped"))
                        self.stop_training_var = False

                        return None

                    model.eval()

                    y_hat = model(X)
                    loss = loss_fn(y_hat, y)
                    
                    test_acc += ((torch.eq(y, y_hat.argmax(dim=1)).sum().item()) / len(y)) * 100

                    test_loss += loss.item()

                test_loss /= len(test_dataloader)
                test_acc /= len(test_dataloader)
        
            self.epoch_list.append(epoch)
            self.train_acc_list.append(train_acc)
            self.train_loss_list.append(train_loss)
            self.test_acc_list.append(test_acc)
            self.test_loss_list.append(test_loss)

            self.root.after(0, self.draw_metrics)
            self.root.after(0, lambda e=epoch, ta=train_acc, te=test_acc:
                            self.training_log(f"Epoch {e}: Train Acc={ta:.2f}, Test Acc={te:.2f}"))

        self.root.after(0, lambda: self.training_log("[INFO] Training Completed"))
        self.root.update_idletasks()

        self.train_button.config(state=tk.NORMAL)
        self.end_train_button.config(state=tk.DISABLED)

        self.root.after(0, lambda: self.training_log("[INFO] Evaluating Model, PLEASE WAIT"))
        self.root.update_idletasks()

        self.micro_metrics, self.macro_metrics, self.y_hat, self.y_pred = self.run_diagnostic(model, test_dataset, task="multiclass", device=device)

        self.root.after(0, lambda: self.open_diagnostic_panel(self.micro_metrics, self.macro_metrics, self.y_hat, self.y_pred, test_dataset, model))

    def load_datasets(self, train_transform, test_transform, train_dir=pathlib.Path, test_dir=pathlib.Path):
        train = datasets.ImageFolder(root=train_dir, transform=train_transform) 
        test = datasets.ImageFolder(root=test_dir, transform=test_transform)

        return train, test

    def load_dataloaders(self, train: torch.utils.data.Dataset, test: torch.utils.data.Dataset, BATCH_SIZE: int):
        train_dataloader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
        test_dataloader = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=False)

        return train_dataloader, test_dataloader

    def stop_training(self):
        self.stop_training_var = True
        self.log("[INFO] Training Stopped")

        self.train_button.config(state=tk.NORMAL)
        self.end_train_button.config(state=tk.DISABLED)

    def start_training(self):
        self.train_button.config(state=tk.DISABLED)
        self.end_train_button.config(state=tk.NORMAL)

        try:
            self.dataset.get()
            self.batch_size.get()
            self.epochs.get()
            self.lr.get()
        except:
            self.log("[ERROR] Invalid Training Parameters")

            self.train_button.config(state=tk.NORMAL)
            self.end_train_button.config(state=tk.DISABLED)

            return None

        if self.model.get() not in self.model_options or self.dataset.get() not in self.dataset_options:
            self.log("[ERROR] Invalid Training Parameters")

            self.train_button.config(state=tk.NORMAL)
            self.end_train_button.config(state=tk.DISABLED)

            return None

        device = "cuda" if torch.cuda.is_available() else "cpu"

        train_dir, test_dir = self.download_dataset(dataset=self.dataset.get())

        train_transform, test_transform = self.load_transforms(self.data_augmentation.get())
        train, test = self.load_datasets(train_transform, test_transform, train_dir, test_dir)
        train_dataloader, test_dataloader = self.load_dataloaders(train, test, self.batch_size.get())

        model = self.create_model(self.model.get(), device)

        optimizer = optim.AdamW(model.parameters(), lr=self.lr.get(), weight_decay=self.MODEL_HYPERPARAMETERS[self.model.get()]["weight_decay"])
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.start_training_thread(self.epochs.get(),
                                   model,
                                   criterion,
                                   optimizer,
                                   train_dataloader,
                                   test_dataloader,
                                   device,
                                   test)
        

    def open_diagnostic_panel(self, micro_metrics, macro_metrics, y_hat, y_pred, test_dataset, model):
        eval_window = tk.Toplevel(self.root)
        eval_window.title("Model Diagnostics Panel")
        eval_window.geometry("1000x600")
        eval_window.resizable(False, False)

        eval_window.columnconfigure(0, weight=1)
        eval_window.columnconfigure(1, weight=1)
        eval_window.rowconfigure(0, weight=1)
        eval_window.rowconfigure(1, weight=1)

        macro_frame = tk.LabelFrame(eval_window, text="Macro Evaluation", padx=10, pady=10)
        macro_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        macro_text = tk.Text(macro_frame, height=10, width=40, state='normal')
        for key, val in macro_metrics.items():
            macro_text.insert(tk.END, f"{key}: {val:.4f}\n")
        macro_text.config(state='disabled')
        macro_text.pack(expand=True, fill=tk.BOTH)

        micro_frame = tk.LabelFrame(eval_window, text="Micro Evaluation", padx=10, pady=10)
        micro_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        micro_text = tk.Text(micro_frame, height=10, width=40, state='normal')
        for key, val in micro_metrics.items():
            micro_text.insert(tk.END, f"{key}: {val:.4f}\n")
        micro_text.config(state='disabled')
        micro_text.pack(expand=True, fill=tk.BOTH)

        info_frame = tk.LabelFrame(eval_window, text="Training Information", padx=10, pady=10)
        info_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        info_text = tk.Text(info_frame, height=10, width=40, state='normal')

        info_text.insert(tk.END, f"Model: {self.model.get()}\n")
        info_text.insert(tk.END, f"Epochs: {self.epochs.get()}\n")
        info_text.insert(tk.END, f"Batch Size: {self.batch_size.get()}\n")
        info_text.insert(tk.END, f"Learning Rate: {self.lr.get()}\n")
        info_text.insert(tk.END, f"Dataset: {self.dataset.get()}\n")
        info_text.insert(tk.END, f"Data Augmentation: {self.data_augmentation.get()}\n")

        info_text.config(state='disabled')
        info_text.pack(expand=True, fill=tk.BOTH)

        export_frame = tk.LabelFrame(eval_window, text="Export Options", padx=10, pady=10)
        export_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        def export_metrics_pdf():
            path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])

            if not path:
                return 
            
            y_prob = self.convert_to_probabilities(y_hat, task='multiclass')
            y_true_array = np.array(test_dataset.targets)

            with PdfPages(path) as pdf:
                fig_text, ax_text = plt.subplots(figsize=(6, 6))
                ax_text.axis("off")
                text = "Training Information:\n"

                text += f"Model: {self.model.get()}\n"
                text += f"Epochs: {self.epochs.get()}\n"
                text += f"Batch Size: {self.batch_size.get()}\n"
                text += f"Learning Rate: {self.lr.get()}\n"
                text += f"Dataset: {self.dataset.get()}\n"
                text += f"Data Augmentation: {self.data_augmentation.get()}\n"

                ax_text.text(0.05, 0.95, text, verticalalignment='top', fontsize=12)
                pdf.savefig(fig_text)
                plt.close(fig_text)
                
                cm = confusion_matrix(y_true_array, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
                disp.plot(ax=ax_cm, cmap=plt.cm.Blues, colorbar=False)
                ax_cm.set_title("Confusion Matrix")
                pdf.savefig(fig_cm)
                plt.close(fig_cm)

                fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                for i in range(len(test_dataset.classes)):
                    binary_true = (y_true_array == i).astype(int)
                    y_scores = y_prob[:, i].cpu().detach().numpy()  
                    fpr, tpr, _ = roc_curve(binary_true, y_scores)
                    roc_auc = auc(fpr, tpr)
                    ax_roc.plot(fpr, tpr, label=f"{test_dataset.classes[i]} (AUC = {roc_auc:.2f})")
                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set_title("ROC Curve")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.legend()
                pdf.savefig(fig_roc)
                plt.close(fig_roc)

                fig_text, ax_text = plt.subplots(figsize=(6, 6))
                ax_text.axis("off")
                text = "Macro Metrics:\n"
                for k, v in macro_metrics.items():
                    text += f"{k}: {v:.4f}\n"
                text += "\nMicro Metrics:\n"
                for k, v in micro_metrics.items():
                    text += f"{k}: {v:.4f}\n"
                ax_text.text(0.05, 0.95, text, verticalalignment='top', fontsize=12)
                pdf.savefig(fig_text)
                plt.close(fig_text)

                self.log(f"[INFO] Saved Model Report To: {path}")

        def export_model():
            path = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PTH Files", "*.pth")])

            if path:
                target_dir_path = pathlib.Path(path)  

                torch.save(obj=model.state_dict(), f=target_dir_path)

                self.log(f"[INFO] Saved Model To: {target_dir_path}")

        tk.Button(export_frame, text="Export Model Evaluation as PDF", command=export_metrics_pdf).pack(pady=5)
        tk.Button(export_frame, text="Download Model Parameters", command=export_model).pack(pady=5)

    def convert_to_probabilities(self, predictions: torch.Tensor, task: str):
        if task == "multiclass":
            return torch.softmax(predictions, dim=1)
        elif task == "binary":
            return torch.sigmoid(predictions, dim=1)

    def make_predictions(self, model, dataset, device, BATCH_SIZE=32):
        model.eval()
    
        with torch.inference_mode():
            predictions = torch.tensor([], device=device)
            
            for X, _ in torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE):
                X = X.to(device)
                
                prediction = model(X)
                predictions = torch.cat((predictions, prediction), dim=0)

        return predictions
        
    def run_diagnostic(self, model, dataset, task, device, BATCH_SIZE=32):
        macro_accuracy = Accuracy(task=task, num_classes=len(dataset.classes), average="macro").to(device)
        macro_precision = Precision(task=task, num_classes=len(dataset.classes), average="macro").to(device)
        macro_recall = Recall(task=task, num_classes=len(dataset.classes), average="macro").to(device)
        macro_f1_score = F1Score(task=task, num_classes=len(dataset.classes), average="macro").to(device)
        macro_auroc = AUROC(task=task, num_classes=len(dataset.classes)).to(device)
    
        micro_accuracy = Accuracy(task=task, num_classes=len(dataset.classes), average="micro").to(device)
        micro_precision = Precision(task=task, num_classes=len(dataset.classes), average="micro").to(device)
        micro_recall = Recall(task=task, num_classes=len(dataset.classes), average="micro").to(device)
        micro_f1_score = F1Score(task=task, num_classes=len(dataset.classes), average="micro").to(device)
        micro_auroc = AUROC(task=task, num_classes=len(dataset.classes)).to(device)

        model.eval()

        y_hat = self.make_predictions(model=model, dataset=dataset, device=device, BATCH_SIZE=BATCH_SIZE)
        y_pred = self.convert_to_probabilities(predictions=y_hat, task=task).argmax(dim=1)

        macro_metrics = {"Accuracy": macro_accuracy(y_pred, torch.tensor(dataset.targets, device=device)).item(), 
                         "Precision": macro_precision(y_pred, torch.tensor(dataset.targets, device=device)).item(), 
                         "Recall": macro_recall(y_pred, torch.tensor(dataset.targets, device=device)).item(),
                         "F1-Score": macro_f1_score(y_pred, torch.tensor(dataset.targets, device=device)).item(),
                         "AUROC": macro_auroc(y_hat, torch.tensor(dataset.targets, device=device)).item()}
        
        micro_metrics = {"Accuracy": micro_accuracy(y_pred, torch.tensor(dataset.targets, device=device)).item(), 
                         "Precision": micro_precision(y_pred, torch.tensor(dataset.targets, device=device)).item(), 
                         "Recall": micro_recall(y_pred, torch.tensor(dataset.targets, device=device)).item(),
                         "F1-Score": micro_f1_score(y_pred, torch.tensor(dataset.targets, device=device)).item(),
                         "AUROC": micro_auroc(y_hat, torch.tensor(dataset.targets, device=device)).item()}
        
        return micro_metrics, macro_metrics, y_hat, y_pred

    def open_training_window(self):
        train_win = Toplevel(self.root)
        train_win.title("Training Monitor")
        train_win.geometry("1000x600")
        train_win.resizable(False, False)

        left_frame = tk.Frame(train_win, width=500, height=600)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        right_frame = tk.Frame(train_win, width=500, height=600)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        console = tk.Text(left_frame, width=60, state=tk.DISABLED, bg="#111", fg="#0f0")
        console.pack(fill=tk.BOTH, expand=True)

        fig = Figure(figsize=(6, 6), dpi=100)
        axs = [fig.add_subplot(221), fig.add_subplot(222),
            fig.add_subplot(223), fig.add_subplot(224)]

        axs[0].set_title("Train Accuracy")
        axs[1].set_title("Train Loss")
        axs[2].set_title("Test Accuracy")
        axs[3].set_title("Test Loss")

        for ax in axs:
            ax.plot([], []) 
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.07, hspace=0.4, wspace=0.3)

        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.training_console = console
        self.training_fig = fig
        self.training_axes = axs
        self.training_canvas = canvas

    def training_log(self, message):
        if hasattr(self, "training_console"):
            self.training_console.config(state=tk.NORMAL)
            self.training_console.insert(tk.END, message + "\n")
            self.training_console.see(tk.END)
            self.training_console.config(state=tk.DISABLED)

    def start_training_thread(self, 
                              epochs,
                              model,
                              criterion,
                              optimizer,
                              train_dataloader,
                              test_dataloader,
                              device,
                              test_dataset):
        
        self.open_training_window()
        thread = threading.Thread(target=self.training_loop,
                                  args=(epochs, model, criterion, optimizer, train_dataloader, test_dataloader, device, test_dataset))
        thread.start()

    def draw_metrics(self):
        axs = self.training_axes

        axs[0].clear()
        axs[0].set_title("Train Accuracy")
        axs[0].plot(self.epoch_list, self.train_acc_list, label="Train Acc", color="blue")
        axs[0].legend()

        axs[1].clear()
        axs[1].set_title("Train Loss")
        axs[1].plot(self.epoch_list, self.train_loss_list, label="Train Loss", color="red")
        axs[1].legend()

        axs[2].clear()
        axs[2].set_title("Test Accuracy")
        axs[2].plot(self.epoch_list, self.test_acc_list, label="Test Acc", color="green")
        axs[2].legend()

        axs[3].clear()
        axs[3].set_title("Test Loss")
        axs[3].plot(self.epoch_list, self.test_loss_list, label="Test Loss", color="orange")
        axs[3].legend()

        self.training_canvas.draw()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = BaseApp(root)
    root.mainloop()
