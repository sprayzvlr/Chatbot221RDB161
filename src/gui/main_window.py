import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import os
import sys
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.chatbot import Chatbot

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Chatbot")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # Set icon if available
        try:
            self.root.iconbitmap("resources/chatbot_icon.ico")
        except:
            pass
            
        # Configure style
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", background="#4a7abc", foreground="black")
        self.style.configure("Chat.TFrame", background="#ffffff")
        
        # Create the chatbot instance
        self.chatbot = Chatbot()
        self.model_loaded = False
        
        # Create main container
        self.main_container = ttk.Frame(self.root, style="TFrame")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create chat header
        self.header_frame = ttk.Frame(self.main_container, style="TFrame")
        self.header_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.header_label = ttk.Label(
            self.header_frame, 
            text="AI Chatbot", 
            font=("Arial", 18, "bold"), 
            background="#f0f0f0"
        )
        self.header_label.pack(side=tk.LEFT, padx=5)
        
        # Create status indicator
        self.status_frame = ttk.Frame(self.header_frame, style="TFrame")
        self.status_frame.pack(side=tk.RIGHT, padx=5)
        
        self.status_label = ttk.Label(
            self.status_frame, 
            text="Status: ", 
            background="#f0f0f0"
        )
        self.status_label.pack(side=tk.LEFT)
        
        self.status_indicator = ttk.Label(
            self.status_frame, 
            text="Not Loaded", 
            foreground="red", 
            background="#f0f0f0"
        )
        self.status_indicator.pack(side=tk.LEFT)
        
        # Create chat area
        self.chat_frame = ttk.Frame(self.main_container, style="Chat.TFrame")
        self.chat_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame, 
            wrap=tk.WORD, 
            font=("Arial", 10),
            background="#ffffff",
            state="disabled"
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create input area
        self.input_frame = ttk.Frame(self.main_container, style="TFrame")
        self.input_frame.pack(fill=tk.X, pady=5)
        
        self.input_field = ttk.Entry(
            self.input_frame, 
            font=("Arial", 10)
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_field.bind("<Return>", self.send_message)
        
        self.send_button = ttk.Button(
            self.input_frame, 
            text="Send", 
            command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # Create button area
        self.button_frame = ttk.Frame(self.main_container, style="TFrame")
        self.button_frame.pack(fill=tk.X, pady=5)
        
        self.load_button = ttk.Button(
            self.button_frame, 
            text="Load Model", 
            command=self.load_model
        )
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.stats_button = ttk.Button(
            self.button_frame, 
            text="Statistics", 
            command=self.open_statistics_window
        )
        self.stats_button.pack(side=tk.LEFT, padx=5)
        
        self.train_button = ttk.Button(
            self.button_frame, 
            text="Training", 
            command=self.open_training_window
        )
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(
            self.button_frame, 
            text="Clear Chat", 
            command=self.clear_chat
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Initialize display
        self.display_welcome_message()
        
        # Create windows dictionary
        self.windows = {}
        
    def display_welcome_message(self):
        """Display welcome message in chat"""
        self.chat_display.configure(state="normal")
        self.chat_display.insert(tk.END, "Welcome to AI Chatbot!\n\n")
        self.chat_display.insert(tk.END, "Please load the model to start chatting.\n\n")
        self.chat_display.configure(state="disabled")
        
    def load_model(self):
        """Load the chatbot model"""
        # Disable buttons during loading
        self.load_button.configure(state="disabled")
        self.status_indicator.configure(text="Loading...", foreground="orange")
        self.root.update()
        
        # Load model in a separate thread
        thread = threading.Thread(target=self._load_model_thread)
        thread.daemon = True
        thread.start()
        
    def _load_model_thread(self):
        """Load model in a separate thread"""
        try:
            # Load the model
            success = self.chatbot.load()
            
            # Update UI
            self.root.after(0, self._update_ui_after_loading, success)
        except Exception as e:
            self.root.after(0, self._update_ui_after_loading, False, str(e))
            
    def _update_ui_after_loading(self, success, error_msg=None):
        """Update UI after model loading"""
        if success:
            self.model_loaded = True
            self.status_indicator.configure(text="Loaded", foreground="green")
            self.load_button.configure(state="normal", text="Reload Model")
            
            # Add system message
            self.chat_display.configure(state="normal")
            self.chat_display.insert(tk.END, "System: Model loaded successfully. You can start chatting now!\n\n")
            self.chat_display.see(tk.END)
            self.chat_display.configure(state="disabled")
        else:
            self.model_loaded = False
            self.status_indicator.configure(text="Failed", foreground="red")
            self.load_button.configure(state="normal")
            
            # Add error message
            self.chat_display.configure(state="normal")
            self.chat_display.insert(tk.END, f"System: Failed to load model. {error_msg if error_msg else ''}\n\n")
            self.chat_display.see(tk.END)
            self.chat_display.configure(state="disabled")
            
    def send_message(self, event=None):
        """Send a message to the chatbot"""
        # Get the user message
        user_message = self.input_field.get().strip()
        
        # Clear the input field
        self.input_field.delete(0, tk.END)
        
        # If message is empty, do nothing
        if not user_message:
            return
            
        # If model is not loaded, show a message
        if not self.model_loaded:
            messagebox.showwarning("Model Not Loaded", "Please load the model first.")
            return
            
        # Display user message
        self.chat_display.configure(state="normal")
        self.chat_display.insert(tk.END, f"You: {user_message}\n")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state="disabled")
        
        # Get and display chatbot response in a separate thread
        thread = threading.Thread(target=self._get_response_thread, args=(user_message,))
        thread.daemon = True
        thread.start()
        
    def _get_response_thread(self, user_message):
        """Get chatbot response in a separate thread"""
        try:
            # Get response from chatbot
            response_data = self.chatbot.get_response(user_message)
            
            # Update UI with the response
            self.root.after(0, self._update_ui_with_response, response_data)
        except Exception as e:
            self.root.after(0, self._update_ui_with_error, str(e))
            
    def _update_ui_with_response(self, response_data):
        """Update UI with chatbot response"""
        # Extract data
        response = response_data['response']
        intent = response_data['intent']
        confidence = response_data['confidence']
        
        # Display response
        self.chat_display.configure(state="normal")
        self.chat_display.insert(tk.END, f"Bot: {response}\n")
        self.chat_display.insert(tk.END, f"[Intent: {intent}, Confidence: {confidence:.2f}]\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state="disabled")
        
    def _update_ui_with_error(self, error_msg):
        """Update UI with error message"""
        self.chat_display.configure(state="normal")
        self.chat_display.insert(tk.END, f"System: Error getting response. {error_msg}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state="disabled")
        
    def clear_chat(self):
        """Clear the chat display"""
        self.chat_display.configure(state="normal")
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.configure(state="disabled")
        
        # Display welcome message
        self.display_welcome_message()
        
    def open_statistics_window(self):
        """Open the statistics window"""
        if 'statistics' in self.windows and self.windows['statistics'].winfo_exists():
            # Window already exists, bring it to front
            self.windows['statistics'].lift()
            return
            
        # Create new window
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Chatbot Statistics")
        stats_window.geometry("800x600")
        stats_window.minsize(800, 600)
        
        # Create statistics window
        from src.gui.statistics_window import StatisticsWindow
        stats_ui = StatisticsWindow(stats_window, self.chatbot)
        
        # Store window reference
        self.windows['statistics'] = stats_window
        
    def open_training_window(self):
        """Open the training window"""
        if 'training' in self.windows and self.windows['training'].winfo_exists():
            # Window already exists, bring it to front
            self.windows['training'].lift()
            return
            
        # Create new window
        train_window = tk.Toplevel(self.root)
        train_window.title("Training and Evaluation")
        train_window.geometry("900x700")
        train_window.minsize(900, 700)
        
        # Create training window
        from src.gui.training_window import TrainingWindow
        train_ui = TrainingWindow(train_window, self.chatbot)
        
        # Store window reference
        self.windows['training'] = train_window
        
    def open_evaluation_window(self):
        """Open the evaluation window"""
        if 'evaluation' in self.windows and self.windows['evaluation'].winfo_exists():
            # Window already exists, bring it to front
            self.windows['evaluation'].lift()
            return
            
        # Create new window
        eval_window = tk.Toplevel(self.root)
        eval_window.title("Model Evaluation")
        eval_window.geometry("800x600")
        eval_window.minsize(800, 600)
        
        # Create evaluation window
        from src.gui.evaluation_window import EvaluationWindow
        eval_ui = EvaluationWindow(eval_window, self.chatbot)
        
        # Store window reference
        self.windows['evaluation'] = eval_window
        
    def save_conversation(self):
        """Save the current conversation"""
        if not self.chatbot.get_conversation_history():
            messagebox.showinfo("Info", "No conversation to save.")
            return
            
        # Save conversation
        success = self.chatbot.save_conversation_history()
        
        if success:
            messagebox.showinfo("Success", "Conversation saved successfully.")
        else:
            messagebox.showerror("Error", "Failed to save conversation.")
            
    def exit_application(self):
        """Exit the application"""
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.root.destroy()
