import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import os
import sys
import json
from matplotlib.figure import Figure

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class StatisticsWindow:
    def __init__(self, root, chatbot):
        self.root = root
        self.chatbot = chatbot
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", background="#4a7abc", foreground="black")
        self.style.configure("TNotebook", background="#f0f0f0")
        self.style.configure("TNotebook.Tab", padding=[10, 5], background="#e0e0e0")
        
        # Create main container
        self.main_container = ttk.Frame(self.root, style="TFrame")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create header
        self.header_frame = ttk.Frame(self.main_container, style="TFrame")
        self.header_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.header_label = ttk.Label(
            self.header_frame, 
            text="Chatbot Statistics", 
            font=("Arial", 16, "bold"), 
            background="#f0f0f0"
        )
        self.header_label.pack(side=tk.LEFT, padx=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create conversation tab
        self.conversation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.conversation_tab, text="Conversation Stats")
        
        # Create performance tab
        self.performance_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.performance_tab, text="Model Performance")
        
        # Create intent distribution tab
        self.intent_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.intent_tab, text="Intent Distribution")
        
        # Create action buttons
        self.button_frame = ttk.Frame(self.main_container, style="TFrame")
        self.button_frame.pack(fill=tk.X, pady=10)
        
        self.refresh_button = ttk.Button(
            self.button_frame, 
            text="Refresh", 
            command=self.refresh_statistics
        )
        self.refresh_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(
            self.button_frame, 
            text="Clear History", 
            command=self.clear_history
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(
            self.button_frame, 
            text="Save Statistics", 
            command=self.save_statistics
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Initialize the tabs
        self.initialize_conversation_tab()
        self.initialize_performance_tab()
        self.initialize_intent_tab()
        
        # Initial statistics refresh
        self.refresh_statistics()
        
    def initialize_conversation_tab(self):
        """Initialize conversation statistics tab"""
        # Create frames for graphs
        self.conv_top_frame = ttk.Frame(self.conversation_tab)
        self.conv_top_frame.pack(fill=tk.X, expand=True, pady=5)
        
        self.conv_bottom_frame = ttk.Frame(self.conversation_tab)
        self.conv_bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create figure for conversation count
        self.conv_count_fig = Figure(figsize=(6, 4), dpi=100)
        self.conv_count_canvas = FigureCanvasTkAgg(self.conv_count_fig, self.conv_top_frame)
        self.conv_count_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create figure for response time
        self.response_time_fig = Figure(figsize=(6, 4), dpi=100)
        self.response_time_canvas = FigureCanvasTkAgg(self.response_time_fig, self.conv_top_frame)
        self.response_time_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create table for conversation data
        self.conversation_table_frame = ttk.Frame(self.conv_bottom_frame)
        self.conversation_table_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create table headers
        self.conv_headers = ['User Input', 'Bot Response', 'Intent', 'Confidence', 'Response Time (s)']
        self.conv_table = ttk.Treeview(
            self.conversation_table_frame, 
            columns=self.conv_headers, 
            show='headings'
        )
        
        # Configure table columns
        for header in self.conv_headers:
            self.conv_table.heading(header, text=header)
            if header in ['User Input', 'Bot Response']:
                self.conv_table.column(header, width=200, stretch=True)
            else:
                self.conv_table.column(header, width=100, stretch=True)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(self.conversation_table_frame, orient="vertical", command=self.conv_table.yview)
        h_scrollbar = ttk.Scrollbar(self.conversation_table_frame, orient="horizontal", command=self.conv_table.xview)
        self.conv_table.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack table and scrollbars
        self.conv_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def initialize_performance_tab(self):
        """Initialize performance statistics tab"""
        # Create frames for graphs
        self.perf_top_frame = ttk.Frame(self.performance_tab)
        self.perf_top_frame.pack(fill=tk.X, expand=True, pady=5)
        
        self.perf_bottom_frame = ttk.Frame(self.performance_tab)
        self.perf_bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create figure for accuracy/confidence
        self.accuracy_fig = Figure(figsize=(6, 4), dpi=100)
        self.accuracy_canvas = FigureCanvasTkAgg(self.accuracy_fig, self.perf_top_frame)
        self.accuracy_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create figure for response times
        self.perf_time_fig = Figure(figsize=(6, 4), dpi=100)
        self.perf_time_canvas = FigureCanvasTkAgg(self.perf_time_fig, self.perf_top_frame)
        self.perf_time_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create metrics summary
        self.metrics_frame = ttk.LabelFrame(self.perf_bottom_frame, text="Performance Metrics")
        self.metrics_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create metrics grid
        metrics = [
            ("Total Conversations:", "total_conv", "0"), 
            ("Average Confidence:", "avg_conf", "0.00"), 
            ("Average Response Time:", "avg_time", "0.00 s"),
            ("Most Common Intent:", "common_intent", "None"),
            ("Least Common Intent:", "least_intent", "None")
        ]
        
        # Create labels for metrics
        self.metric_labels = {}
        for i, (label_text, label_name, default_value) in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            # Create label frame
            frame = ttk.Frame(self.metrics_frame)
            frame.grid(row=row, column=col, padx=10, pady=5, sticky="nsew")
            
            # Create label and value
            label = ttk.Label(frame, text=label_text, font=("Arial", 10, "bold"))
            label.pack(anchor="w", pady=2)
            
            value = ttk.Label(frame, text=default_value, font=("Arial", 10))
            value.pack(anchor="w", pady=2)
            
            # Store label reference
            self.metric_labels[label_name] = value
        
        # Configure grid weights
        for i in range(2):
            self.metrics_frame.grid_rowconfigure(i, weight=1)
        for i in range(3):
            self.metrics_frame.grid_columnconfigure(i, weight=1)
        
    def initialize_intent_tab(self):
        """Initialize intent distribution tab"""
        # Create frames for graphs
        self.intent_left_frame = ttk.Frame(self.intent_tab)
        self.intent_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5)
        
        self.intent_right_frame = ttk.Frame(self.intent_tab)
        self.intent_right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=5)
        
        # Create figure for intent distribution pie chart
        self.intent_pie_fig = Figure(figsize=(6, 5), dpi=100)
        self.intent_pie_canvas = FigureCanvasTkAgg(self.intent_pie_fig, self.intent_left_frame)
        self.intent_pie_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create figure for intent confidence bar chart
        self.intent_bar_fig = Figure(figsize=(6, 5), dpi=100)
        self.intent_bar_canvas = FigureCanvasTkAgg(self.intent_bar_fig, self.intent_right_frame)
        self.intent_bar_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def refresh_statistics(self):
        """Refresh all statistics"""
        # Get conversation history
        history = self.chatbot.get_conversation_history()
        
        if not history:
            self.clear_all_plots()
            messagebox.showinfo("Info", "No conversation history available.")
            return
        
        # Get statistics
        stats = self.chatbot.get_intent_statistics()
        
        # Update conversation tab
        self.update_conversation_plots(history, stats)
        self.update_conversation_table(history)
        
        # Update performance tab
        self.update_performance_plots(history, stats)
        self.update_metrics_summary(stats)
        
        # Update intent tab
        self.update_intent_plots(stats)
        
    def update_conversation_plots(self, history, stats):
        """Update plots in the conversation tab"""
        # Clear previous plots
        self.conv_count_fig.clear()
        self.response_time_fig.clear()
        
        # Extract data for conversation count over time
        timestamps = list(range(1, len(history) + 1))
        intents = [exchange['intent'] for exchange in history]
        
        # Create conversation count plot
        ax1 = self.conv_count_fig.add_subplot(111)
        ax1.plot(timestamps, intents, 'bo-')
        ax1.set_xlabel('Conversation Number')
        ax1.set_ylabel('Intent')
        ax1.set_title('Conversation Intents Over Time')
        
        # Extract data for response time
        response_times = [exchange['response_time'] for exchange in history]
        
        # Create response time plot
        ax2 = self.response_time_fig.add_subplot(111)
        ax2.plot(timestamps, response_times, 'ro-')
        ax2.set_xlabel('Conversation Number')
        ax2.set_ylabel('Response Time (s)')
        ax2.set_title('Response Time Over Conversations')
        
        # Adjust layout and redraw
        self.conv_count_fig.tight_layout()
        self.response_time_fig.tight_layout()
        self.conv_count_canvas.draw()
        self.response_time_canvas.draw()
        
    def update_conversation_table(self, history):
        """Update the conversation history table"""
        # Clear previous data
        for item in self.conv_table.get_children():
            self.conv_table.delete(item)
            
        # Add new data
        for i, exchange in enumerate(history):
            values = (
                exchange['user'],
                exchange['bot'],
                exchange['intent'],
                f"{exchange['confidence']:.2f}",
                f"{exchange['response_time']:.4f}"
            )
            self.conv_table.insert('', tk.END, values=values)
        
    def update_performance_plots(self, history, stats):
        """Update plots in the performance tab"""
        # Clear previous plots
        self.accuracy_fig.clear()
        self.perf_time_fig.clear()
        
        # Extract confidence data
        intents = list(stats['avg_confidences'].keys())
        confidences = list(stats['avg_confidences'].values())
        
        # Create confidence bar chart
        ax1 = self.accuracy_fig.add_subplot(111)
        bars = ax1.bar(intents, confidences, color='skyblue')
        ax1.set_xlabel('Intent')
        ax1.set_ylabel('Average Confidence')
        ax1.set_title('Average Confidence by Intent')
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom')
        
        # Extract response time data
        timestamps = list(range(1, len(history) + 1))
        response_times = [exchange['response_time'] for exchange in history]
        
        # Calculate moving average if enough data points
        if len(response_times) > 3:
            window_size = min(3, len(response_times))
            moving_avg = np.convolve(response_times, np.ones(window_size)/window_size, mode='valid')
            moving_avg_x = list(range(window_size, len(response_times) + 1))
        else:
            moving_avg = response_times
            moving_avg_x = timestamps
        
        # Create response time plot
        ax2 = self.perf_time_fig.add_subplot(111)
        ax2.plot(timestamps, response_times, 'ro-', label='Response Time')
        if len(response_times) > 3:
            ax2.plot(moving_avg_x, moving_avg, 'b-', label='Moving Average')
        ax2.set_xlabel('Conversation Number')
        ax2.set_ylabel('Response Time (s)')
        ax2.set_title('Response Time Performance')
        ax2.legend()
        
        # Adjust layout and redraw
        self.accuracy_fig.tight_layout()
        self.perf_time_fig.tight_layout()
        self.accuracy_canvas.draw()
        self.perf_time_canvas.draw()
        
    def update_metrics_summary(self, stats):
        """Update the metrics summary"""
        # Update metrics labels
        self.metric_labels['total_conv'].config(text=str(stats['total_exchanges']))
        
        # Calculate average confidence across all intents
        avg_conf = sum(stats['avg_confidences'].values()) / len(stats['avg_confidences'])
        self.metric_labels['avg_conf'].config(text=f"{avg_conf:.2f}")
        
        # Update average response time
        self.metric_labels['avg_time'].config(text=f"{stats['avg_response_time']:.4f} s")
        
        # Find most and least common intents
        intent_counts = stats['intent_counts']
        if intent_counts:
            most_common = max(intent_counts.items(), key=lambda x: x[1])
            least_common = min(intent_counts.items(), key=lambda x: x[1])
            
            self.metric_labels['common_intent'].config(text=f"{most_common[0]} ({most_common[1]} times)")
            self.metric_labels['least_intent'].config(text=f"{least_common[0]} ({least_common[1]} times)")
        
    def update_intent_plots(self, stats):
        """Update plots in the intent tab"""
        # Clear previous plots
        self.intent_pie_fig.clear()
        self.intent_bar_fig.clear()
        
        # Extract data
        intents = list(stats['intent_counts'].keys())
        counts = list(stats['intent_counts'].values())
        percentages = list(stats['intent_percentages'].values())
        
        # Create pie chart
        ax1 = self.intent_pie_fig.add_subplot(111)
        wedges, texts, autotexts = ax1.pie(
            counts, 
            labels=intents, 
            autopct='%1.1f%%',
            startangle=90, 
            shadow=True
        )
        ax1.set_title('Intent Distribution')
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Create bar chart for confidences
        ax2 = self.intent_bar_fig.add_subplot(111)
        confidences = list(stats['avg_confidences'].values())
        bars = ax2.bar(intents, confidences, color='lightgreen')
        ax2.set_xlabel('Intent')
        ax2.set_ylabel('Average Confidence')
        ax2.set_title('Average Confidence by Intent')
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom')
        
        # Adjust layout and redraw
        self.intent_pie_fig.tight_layout()
        self.intent_bar_fig.tight_layout()
        self.intent_pie_canvas.draw()
        self.intent_bar_canvas.draw()
        
    def clear_all_plots(self):
        """Clear all plots"""
        # Clear conversation tab plots
        self.conv_count_fig.clear()
        self.response_time_fig.clear()
        self.conv_count_canvas.draw()
        self.response_time_canvas.draw()
        
        # Clear performance tab plots
        self.accuracy_fig.clear()
        self.perf_time_fig.clear()
        self.accuracy_canvas.draw()
        self.perf_time_canvas.draw()
        
        # Clear intent tab plots
        self.intent_pie_fig.clear()
        self.intent_bar_fig.clear()
        self.intent_pie_canvas.draw()
        self.intent_bar_canvas.draw()
        
        # Clear table
        for item in self.conv_table.get_children():
            self.conv_table.delete(item)
            
        # Reset metrics
        for label in self.metric_labels.values():
            label.config(text="0")
        
    def clear_history(self):
        """Clear conversation history"""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear the conversation history?"):
            self.chatbot.clear_conversation_history()
            self.clear_all_plots()
            messagebox.showinfo("Info", "Conversation history cleared.")
        
    def save_statistics(self):
        """Save statistics to a file"""
        # Get conversation history and statistics
        history = self.chatbot.get_conversation_history()
        
        if not history:
            messagebox.showinfo("Info", "No statistics to save.")
            return
        
        stats = self.chatbot.get_intent_statistics()
        
        # Create a statistics object
        statistics = {
            'conversation_history': history,
            'statistics': stats,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # Save to file
        filename = f"logs/statistics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, indent=4)
            messagebox.showinfo("Success", f"Statistics saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save statistics: {str(e)}")
