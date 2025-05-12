#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import argparse

# Import GUI
from src.gui.main_window import MainWindow
from src.chatbot import Chatbot

def setup_environment():
    """Set up the environment before launching the application"""
    # Create necessary directories if they don't exist
    dirs = ['data', 'models', 'logs']
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AI Chatbot with TensorFlow')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--model', type=str, help='Path to model file to load at startup')
    parser.add_argument('--no-gui', action='store_true', help='Run in console mode without GUI')
    
    return parser.parse_args()

def console_mode():
    """Run chatbot in console mode"""
    print("=== AI Chatbot (Console Mode) ===")
    print("Type 'exit', 'quit', or Ctrl+C to exit")
    print("Loading model...")
    
    chatbot = Chatbot()
    success = chatbot.load()
    
    if not success:
        print("Failed to load model. Exiting.")
        return
    
    print("Model loaded successfully!")
    print("You can start chatting now.\n")
    
    try:
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Chatbot: Goodbye!")
                break
            
            response = chatbot.get_response(user_input)
            print(f"Chatbot: {response['response']}")
            print(f"[Intent: {response['intent']}, Confidence: {response['confidence']:.2f}]")
            print()
            
    except KeyboardInterrupt:
        print("\nExiting chatbot...")
    except Exception as e:
        print(f"Error: {str(e)}")

def gui_mode(args):
    """Run chatbot with GUI"""
    # Set up the main application window
    root = tk.Tk()
    root.title("AI Chatbot")
    
    # Set window icon if available
    try:
        root.iconbitmap("resources/chatbot_icon.ico")
    except:
        pass
    
    # Set default window size and position
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 800
    window_height = 600
    
    # Center the window
    position_x = int((screen_width - window_width) / 2)
    position_y = int((screen_height - window_height) / 2)
    
    root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
    root.minsize(800, 600)
    
    # Set theme and style
    style = ttk.Style()
    style.theme_use('clam')  # Other options: 'alt', 'default', 'classic'
    
    # Debug info
    if args.debug:
        print(f"Python version: {sys.version}")
        print(f"Tkinter version: {tk.TkVersion}")
        print(f"Available themes: {style.theme_names()}")
    
    # Create the main application window
    app = MainWindow(root)
    
    # Auto-load model if specified
    if args.model:
        # Schedule model loading after GUI is fully initialized
        root.after(1000, lambda: load_model(app, args.model))
    
    # Start the main event loop
    root.mainloop()

def load_model(app, model_path):
    """Load the specified model after GUI initialization"""
    try:
        # Override the default model path
        app.chatbot.model_path = model_path
        
        # Load the model
        success = app.chatbot.load()
        
        if success:
            # Update UI to reflect loaded state
            app.status_indicator.configure(text="Loaded", foreground="green")
            app.load_button.configure(text="Reload Model")
            
            # Add system message
            app.chat_display.configure(state="normal")
            app.chat_display.insert(tk.END, f"System: Model loaded from {model_path}\n\n")
            app.chat_display.see(tk.END)
            app.chat_display.configure(state="disabled")
            
            # Set flag
            app.model_loaded = True
            
        else:
            # Show error message
            messagebox.showerror("Error", f"Failed to load model from {model_path}")
            
    except Exception as e:
        messagebox.showerror("Error", f"Error loading model: {str(e)}")

def main():
    """Main application entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up environment
    setup_environment()
    
    # Choose mode based on arguments
    if args.no_gui:
        console_mode()
    else:
        gui_mode(args)

if __name__ == "__main__":
    main()
