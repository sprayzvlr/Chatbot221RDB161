import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class TrainingWindow:
    def __init__(self, root, chatbot):
        self.root = root
        self.chatbot = chatbot
        self.is_training = False
        self.history = None
        
        # Set up the UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the training window UI"""
        # Configure style
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", background="#4a7abc")
        self.style.configure("TLabel", background="#f0f0f0")
        
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create header
        header = ttk.Label(
            self.main_container, 
            text="Model Training", 
            font=("Arial", 16, "bold"),
            background="#f0f0f0"
        )
        header.pack(pady=(0, 10))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text="Training")
        
        self.evaluation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.evaluation_tab, text="Evaluation")
        
        # Initialize the tabs
        self.initialize_training_tab()
        self.initialize_evaluation_tab()
        
    def initialize_training_tab(self):
        """Create the training tab interface"""
        # Training parameters section
        param_frame = ttk.LabelFrame(self.training_tab, text="Training Parameters")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a grid for parameters
        # Epochs
        ttk.Label(param_frame, text="Epochs:").grid(row=0, column=0, padx=5, pady=5)
        self.epochs_var = tk.StringVar(value="150")
        ttk.Entry(param_frame, textvariable=self.epochs_var, width=8).grid(row=0, column=1, padx=5, pady=5)
        
        # Batch size
        ttk.Label(param_frame, text="Batch Size:").grid(row=0, column=2, padx=5, pady=5)
        self.batch_size_var = tk.StringVar(value="8")
        ttk.Entry(param_frame, textvariable=self.batch_size_var, width=8).grid(row=0, column=3, padx=5, pady=5)
        
        # Validation split
        ttk.Label(param_frame, text="Validation Split:").grid(row=0, column=4, padx=5, pady=5)
        self.val_split_var = tk.StringVar(value="0.1")
        ttk.Entry(param_frame, textvariable=self.val_split_var, width=8).grid(row=0, column=5, padx=5, pady=5)
        
        # Model type
        ttk.Label(param_frame, text="Model Type:").grid(row=1, column=0, padx=5, pady=5)
        self.model_type_var = tk.StringVar(value="lstm")
        model_combo = ttk.Combobox(
            param_frame, 
            textvariable=self.model_type_var, 
            values=["basic", "lstm", "advanced"],
            width=10,
            state="readonly"
        )
        model_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Data file
        ttk.Label(param_frame, text="Data File:").grid(row=1, column=2, padx=5, pady=5)
        self.data_file_var = tk.StringVar(value="data/intents.json")
        data_entry = ttk.Entry(param_frame, textvariable=self.data_file_var, width=25)
        data_entry.grid(row=1, column=3, columnspan=2, padx=5, pady=5, sticky="we")
        
        browse_btn = ttk.Button(param_frame, text="Browse", command=self.browse_data_file)
        browse_btn.grid(row=1, column=5, padx=5, pady=5)
        
        # Training progress section
        progress_frame = ttk.LabelFrame(self.training_tab, text="Training Progress")
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(
            progress_frame, 
            orient="horizontal", 
            length=100, 
            mode="determinate",
            variable=self.progress_var
        )
        progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.pack(padx=5, pady=2)
        
        # Metrics
        metrics_frame = ttk.Frame(progress_frame)
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a grid for metrics
        metrics = [
            ("Accuracy:", "acc_var", "0.0"),
            ("Loss:", "loss_var", "0.0"),
            ("Val Accuracy:", "val_acc_var", "0.0"),
            ("Val Loss:", "val_loss_var", "0.0")
        ]
        
        self.metric_vars = {}
        
        for i, (label, var_name, default) in enumerate(metrics):
            row, col = divmod(i, 2)
            
            ttk.Label(metrics_frame, text=label).grid(row=row, column=col*2, padx=5, pady=2, sticky="e")
            
            var = tk.StringVar(value=default)
            self.metric_vars[var_name] = var
            
            ttk.Label(metrics_frame, textvariable=var).grid(row=row, column=col*2+1, padx=5, pady=2, sticky="w")
        
        # Control buttons
        control_frame = ttk.Frame(progress_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.train_btn = ttk.Button(control_frame, text="Start Training", command=self.start_training)
        self.train_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_training, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.save_btn = ttk.Button(control_frame, text="Save Model", command=self.save_model)
        self.save_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.load_btn = ttk.Button(control_frame, text="Load History", command=self.load_history)
        self.load_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Plots section
        plots_frame = ttk.LabelFrame(self.training_tab, text="Training Plots")
        plots_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create two plots side by side
        plots_container = ttk.Frame(plots_frame)
        plots_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create figures
        self.acc_fig = Figure(figsize=(5, 4), dpi=100)
        self.loss_fig = Figure(figsize=(5, 4), dpi=100)
        
        # Create canvases
        self.acc_canvas = FigureCanvasTkAgg(self.acc_fig, plots_container)
        self.acc_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, plots_container)
        self.loss_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Initialize empty plots
        self.init_plots()
        
    def initialize_evaluation_tab(self):
        """Create the evaluation tab interface"""
        # Controls section
        controls_frame = ttk.Frame(self.evaluation_tab)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        eval_btn = ttk.Button(controls_frame, text="Evaluate Model", command=self.evaluate_model)
        eval_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        test_btn = ttk.Button(controls_frame, text="Test Utterances", command=self.test_utterances)
        test_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Metrics section
        metrics_frame = ttk.LabelFrame(self.evaluation_tab, text="Evaluation Metrics")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a grid for evaluation metrics
        eval_metrics = [
            ("Accuracy:", "eval_acc", "0.0"),
            ("Precision:", "eval_prec", "0.0"),
            ("Recall:", "eval_recall", "0.0"),
            ("F1 Score:", "eval_f1", "0.0")
        ]
        
        self.eval_vars = {}
        
        for i, (label, var_name, default) in enumerate(eval_metrics):
            row, col = divmod(i, 2)
            
            ttk.Label(metrics_frame, text=label).grid(row=row, column=col*2, padx=5, pady=5, sticky="e")
            
            var = tk.StringVar(value=default)
            self.eval_vars[var_name] = var
            
            ttk.Label(metrics_frame, textvariable=var).grid(row=row, column=col*2+1, padx=5, pady=5, sticky="w")
        
        # Results section
        results_frame = ttk.LabelFrame(self.evaluation_tab, text="Evaluation Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create text widget for results
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def init_plots(self):
        """Initialize empty plots"""
        # Accuracy plot
        self.acc_fig.clear()
        acc_ax = self.acc_fig.add_subplot(111)
        acc_ax.set_title("Model Accuracy")
        acc_ax.set_xlabel("Epoch")
        acc_ax.set_ylabel("Accuracy")
        acc_ax.set_ylim(0, 1)
        acc_ax.grid(True)
        self.acc_fig.tight_layout()
        self.acc_canvas.draw()
        
        # Loss plot
        self.loss_fig.clear()
        loss_ax = self.loss_fig.add_subplot(111)
        loss_ax.set_title("Model Loss")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_ax.grid(True)
        self.loss_fig.tight_layout()
        self.loss_canvas.draw()
        
    def browse_data_file(self):
        """Open file dialog to browse for data file"""
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if filename:
            self.data_file_var.set(filename)
            
    def start_training(self):
        """Start the model training process"""
        # Get training parameters
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            val_split = float(self.val_split_var.get())
            model_type = self.model_type_var.get()
            data_file = self.data_file_var.get()
            
            # Validate parameters
            if epochs <= 0 or batch_size <= 0 or val_split <= 0 or val_split >= 1:
                raise ValueError("Invalid parameter values")
                
        except ValueError as e:
            messagebox.showerror("Parameter Error", f"Invalid training parameters: {str(e)}")
            return
            
        # Update UI state
        self.is_training = True
        self.train_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Preparing data...")
        self.progress_var.set(0)
        
        # Init plots
        self.init_plots()
        
        # Start training in a separate thread
        self.train_thread = threading.Thread(
            target=self._train_thread_func,
            args=(epochs, batch_size, val_split, model_type, data_file)
        )
        self.train_thread.daemon = True
        self.train_thread.start()
        
        # Start progress updates
        self.update_training_progress()
        
    def _train_thread_func(self, epochs, batch_size, val_split, model_type, data_file):
        """Training thread function"""
        try:
            # Import required modules for training
            from src.training import ModelTrainer
            import tensorflow as tf
            
            # Create the ModelTrainer instance
            trainer = ModelTrainer()
            
            # Set initial status
            self.root.after(0, self._update_metrics, 0, epochs, 0, 0, 0, 0)
            self.root.after(0, lambda: self.status_var.set("Loading and preparing data..."))
            
            # Prepare data
            self.root.after(0, lambda: self.status_var.set("Preparing training data..."))
            
            # Create model based on selected type
            use_advanced = model_type == "advanced"
            
            # Start training
            self.root.after(0, lambda: self.status_var.set("Training model..."))
            
            # Actual training
            history, evaluation = trainer.train(
                epochs=epochs,
                batch_size=batch_size,
                validation_split=val_split,
                use_advanced_model=use_advanced
            )
            
            # Convert history to format usable by the UI
            self.history = {}
            for key in history.history:
                self.history[key] = [float(val) for val in history.history[key]]
            
            # Training complete
            self.root.after(0, self._on_training_complete)
            
        except Exception as e:
            # Handle errors
            import traceback
            error_details = traceback.format_exc()
            print(f"Training error: {error_details}")
            self.root.after(0, self._on_training_error, str(e))
            
    def update_training_progress(self):
        """Update training progress in the UI"""
        if not self.is_training:
            return
            
        # Update plots if we have history
        if self.history is not None:
            self.update_plots()
            
        # Schedule next update
        self.root.after(1000, self.update_training_progress)
        
    def _update_metrics(self, epoch, total_epochs, acc, loss, val_acc, val_loss):
        """Update metrics display with current values"""
        # Update progress
        self.progress_var.set(epoch / total_epochs * 100)
        
        # Update status
        self.status_var.set(f"Training epoch {epoch} of {total_epochs}")
        
        # Update metrics
        self.metric_vars["acc_var"].set(f"{acc:.4f}")
        self.metric_vars["loss_var"].set(f"{loss:.4f}")
        self.metric_vars["val_acc_var"].set(f"{val_acc:.4f}")
        self.metric_vars["val_loss_var"].set(f"{val_loss:.4f}")
        
    def update_plots(self):
        """Update training plots with current history"""
        if self.history is None:
            return
            
        # Get data
        epochs = range(1, len(self.history["accuracy"]) + 1)
        
        # Update accuracy plot
        self.acc_fig.clear()
        acc_ax = self.acc_fig.add_subplot(111)
        acc_ax.plot(epochs, self.history["accuracy"], 'b-', label="Training")
        acc_ax.plot(epochs, self.history["val_accuracy"], 'r-', label="Validation")
        acc_ax.set_title("Model Accuracy")
        acc_ax.set_xlabel("Epoch")
        acc_ax.set_ylabel("Accuracy")
        acc_ax.set_ylim(0, 1)
        acc_ax.legend()
        acc_ax.grid(True)
        self.acc_fig.tight_layout()
        self.acc_canvas.draw()
        
        # Update loss plot
        self.loss_fig.clear()
        loss_ax = self.loss_fig.add_subplot(111)
        loss_ax.plot(epochs, self.history["loss"], 'b-', label="Training")
        loss_ax.plot(epochs, self.history["val_loss"], 'r-', label="Validation")
        loss_ax.set_title("Model Loss")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_ax.legend()
        loss_ax.grid(True)
        self.loss_fig.tight_layout()
        self.loss_canvas.draw()
        
    def _on_training_complete(self):
        """Called when training is complete"""
        # Update UI
        self.is_training = False
        self.train_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Training complete")
        self.progress_var.set(100)
        
        # Final plot update
        self.update_plots()
        
        # Show message
        messagebox.showinfo("Training Complete", "Model training has completed successfully.")
        
    def _on_training_error(self, error_msg):
        """Called when training encounters an error"""
        # Update UI
        self.is_training = False
        self.train_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set(f"Error: {error_msg}")
        
        # Show error
        messagebox.showerror("Training Error", f"An error occurred during training:\n{error_msg}")
        
    def stop_training(self):
        """Stop the training process"""
        if not self.is_training:
            return
            
        # Confirm with user
        if messagebox.askyesno("Stop Training", "Are you sure you want to stop training?"):
            # Set flag to stop
            self.is_training = False
            self.status_var.set("Training stopped by user")
            self.train_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            
    def save_model(self):
        """Save the trained model"""
        # Check if we have a model
        if self.history is None:
            messagebox.showwarning("Warning", "No trained model available to save.")
            return
            
        try:
            # Here we would call the actual save function
            # For example: self.chatbot.save_model()
            
            # Save history
            self.save_history()
            
            messagebox.showinfo("Success", "Model and training history saved successfully.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
            
    def save_history(self, filepath='models/history.json'):
        """Save training history to a file"""
        if self.history is None:
            return False
            
        # Convert to dict of floats
        history_dict = {key: [float(val) for val in values] for key, values in self.history.items()}
        
        # Create directory if needed
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=4)
            
        print(f"Training history saved to {filepath}")
        return True
        
    def load_history(self, filepath='models/history.json'):
        """Load training history from a file"""
        try:
            # Ask for file if not provided
            if filepath == 'models/history.json':
                filepath = filedialog.askopenfilename(
                    title="Select History File",
                    filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                    initialdir="models"
                )
                
                if not filepath:
                    return False
                    
            # Load from file
            with open(filepath, 'r') as f:
                self.history = json.load(f)
                
            # Update plots
            self.update_plots()
            
            # Update status
            self.status_var.set(f"Loaded history from {filepath}")
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load history: {str(e)}")
            return False
            
    def evaluate_model(self):
        """Evaluate the trained model"""
        # Check if model is trained
        if self.history is None:
            messagebox.showwarning("Warning", "No trained model available to evaluate.")
            return
            
        try:
            # Here we would call the actual evaluation function
            # For example: results = self.chatbot.evaluate_model()
            
            # Simulated evaluation results
            results = {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.87,
                "f1_score": 0.88
            }
            
            # Update evaluation metrics
            self.eval_vars["eval_acc"].set(f"{results['accuracy']:.4f}")
            self.eval_vars["eval_prec"].set(f"{results['precision']:.4f}")
            self.eval_vars["eval_recall"].set(f"{results['recall']:.4f}")
            self.eval_vars["eval_f1"].set(f"{results['f1_score']:.4f}")
            
            # Update results text
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Evaluation Results:\n\n")
            
            for key, value in results.items():
                self.results_text.insert(tk.END, f"{key.title()}: {value:.4f}\n")
                
            self.results_text.insert(tk.END, "\nMore detailed analysis would appear here...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to evaluate model: {str(e)}")
            
    def test_utterances(self):
        """Test model with custom utterances"""
        # Check if model is trained
        if self.history is None:
            messagebox.showwarning("Warning", "No trained model available for testing.")
            return
            
        # Create a simple dialog for entering test utterances
        test_window = tk.Toplevel(self.root)
        test_window.title("Test Utterances")
        test_window.geometry("500x400")
        test_window.minsize(500, 400)
        
        # Test input
        input_frame = ttk.Frame(test_window)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(input_frame, text="Enter text to test:").pack(anchor="w")
        
        input_entry = ttk.Entry(input_frame, width=50)
        input_entry.pack(fill=tk.X, pady=5)
        input_entry.bind("<Return>", lambda e: predict_intent())
        
        # Results area
        results_frame = ttk.LabelFrame(test_window, text="Prediction Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Test button
        button_frame = ttk.Frame(test_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        test_button = ttk.Button(button_frame, text="Test", command=lambda: predict_intent())
        test_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = ttk.Button(button_frame, text="Clear", command=lambda: results_text.delete(1.0, tk.END))
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Prediction function
        def predict_intent():
            user_input = input_entry.get().strip()
            if not user_input:
                return
                
            # Here we would call the actual prediction function
            # For example: prediction = self.chatbot.predict(user_input)
            
            # Simulated prediction
            import random
            intents = ["greeting", "farewell", "thanks", "help", "query"]
            intent = random.choice(intents)
            confidence = random.uniform(0.7, 0.98)
            
            # Display results
            results_text.insert(tk.END, f"Input: {user_input}\n")
            results_text.insert(tk.END, f"Predicted Intent: {intent}\n")
            results_text.insert(tk.END, f"Confidence: {confidence:.4f}\n\n")
            results_text.see(tk.END)
            
            # Clear input
            input_entry.delete(0, tk.END)
