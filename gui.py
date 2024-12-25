import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from stock_data import fetch_stock_data
from model import StockPerformanceModel
from utils import preprocess_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class StockMarketSimulatorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stock Market Simulator")
        self.geometry("900x700")
        self.balance = 1_000_000  # Starting balance
        self.portfolio = {}
        self.configure_styles()
        self.create_widgets()

    def configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")     
        style.configure("TFrame", background="#e0f7fa")
        style.configure("TLabel", background="#e0f7fa", font=("Arial", 12))
        style.configure("TButton", background="#00796b", foreground="#ffffff", font=("Arial", 12, "bold"))
        style.configure("TEntry", font=("Arial", 12))      
        self.configure(background="#004d40")
    
    def create_widgets(self):
        self.header_frame = ttk.Frame(self)
        self.header_frame.pack(pady=10)
        
        self.label = ttk.Label(self.header_frame, text="Welcome to the Stock Market Simulator", font=("Arial", 20, "bold"))
        self.label.pack()

        self.balance_frame = ttk.Frame(self)
        self.balance_frame.pack(pady=10)
        
        self.balance_label = ttk.Label(self.balance_frame, text=f"Balance: ${self.balance}", font=("Arial", 16, "bold"))
        self.balance_label.pack()

        self.input_frame = ttk.Frame(self)
        self.input_frame.pack(pady=10)

        self.ticker_label = ttk.Label(self.input_frame, text="Stock Ticker:")
        self.ticker_label.grid(row=0, column=0, padx=5)
        
        self.ticker_entry = ttk.Entry(self.input_frame)
        self.ticker_entry.grid(row=0, column=1, padx=5)
        
        self.amount_label = ttk.Label(self.input_frame, text="Amount to Invest:")
        self.amount_label.grid(row=1, column=0, padx=5)
        
        self.amount_entry = ttk.Entry(self.input_frame)
        self.amount_entry.grid(row=1, column=1, padx=5)
        
        self.buy_button = ttk.Button(self.input_frame, text="Buy", command=self.buy_stock)
        self.buy_button.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.date_frame = ttk.Frame(self)
        self.date_frame.pack(pady=10)

        self.start_date_label = ttk.Label(self.date_frame, text="Start Date (YYYY-MM-DD):")
        self.start_date_label.grid(row=0, column=0, padx=5)
        
        self.start_date_entry = ttk.Entry(self.date_frame)
        self.start_date_entry.grid(row=0, column=1, padx=5)
        
        self.end_date_label = ttk.Label(self.date_frame, text="End Date (YYYY-MM-DD):")
        self.end_date_label.grid(row=1, column=0, padx=5)
        
        self.end_date_entry = ttk.Entry(self.date_frame)
        self.end_date_entry.grid(row=1, column=1, padx=5)
        
        self.simulate_button = ttk.Button(self.date_frame, text="Simulate", command=self.simulate_performance)
        self.simulate_button.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.portfolio_frame = ttk.Frame(self)
        self.portfolio_frame.pack(pady=10)
        
        self.portfolio_label = ttk.Label(self.portfolio_frame, text="Portfolio:", font=("Arial", 16, "bold"))
        self.portfolio_label.pack()

        self.portfolio_text = tk.Text(self.portfolio_frame, height=10, width=60, font=("Arial", 12))
        self.portfolio_text.pack()

        self.result_frame = ttk.Frame(self)
        self.result_frame.pack(pady=10)
        
        self.result_label = ttk.Label(self.result_frame, text="Simulation Results:", font=("Arial", 16, "bold"))
        self.result_label.pack()

        self.result_text = tk.Text(self.result_frame, height=10, width=60, font=("Arial", 12))
        self.result_text.pack()

        self.graph_frame = ttk.Frame(self)
        self.graph_frame.pack(pady=10)


    def buy_stock(self):
        try:
            ticker = self.ticker_entry.get().upper()
            amount = int(self.amount_entry.get())

            if amount > self.balance:
                messagebox.showerror("Error", "Insufficient funds!")
                return

            self.balance -= amount
            self.balance_label.config(text=f"Balance: ${self.balance}")

            if ticker in self.portfolio:
                self.portfolio[ticker] += amount
            else:
                self.portfolio[ticker] = amount

            self.update_portfolio_display()
            messagebox.showinfo("Success", f"Bought ${amount} of {ticker} stock!")
        except ValueError:
            messagebox.showerror("Error", "Invalid input! Please enter a valid amount.")
    
    def update_portfolio_display(self):
        self.portfolio_text.delete(1.0, tk.END)
        for ticker, amount in self.portfolio.items():
            self.portfolio_text.insert(tk.END, f"{ticker}: ${amount}\n")
    
    def simulate_performance(self):
        try:
            if not self.portfolio:
                messagebox.showerror("Error", "No stocks in portfolio to simulate.")
                return

            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()
            
            self.result_text.delete(1.0, tk.END)
            
            for ticker, amount in self.portfolio.items():
                data = fetch_stock_data(ticker, start_date, end_date)
                if data.empty:
                    messagebox.showerror("Error", f"Failed to fetch stock data for {ticker}.")
                    continue

                # Normalize 'Close' prices for initial investment
                initial_price = data['Close'].iloc[0]
                num_shares = amount / initial_price

                model = StockPerformanceModel()
                model.train(data)
                
                # Predict the price for the end date
                future_price = model.predict(data)
                
                # Calculate returns
                net_return = (future_price - initial_price) * num_shares
                percent_return = (net_return / amount) * 100

                # Display results
                result_str = (
                    f"Stock: {ticker}\n"
                    f"Predicted future price: ${future_price:.2f}\n"
                    f"Initial Investment: ${amount}\n"
                    f"Net Return: ${net_return:.2f}\n"
                    f"Percent Return: {percent_return:.2f}%\n\n"
                )
                self.result_text.insert(tk.END, result_str)

                # Plot the graph
                self.plot_graph(data, future_price)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during simulation: {e}")

    def plot_graph(self, data, future_price):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(data.index, data['Close'], label='Historical Prices')
        ax.axhline(y=future_price, color='r', linestyle='--', label=f'Predicted Price: ${future_price:.2f}')
        ax.set_title('Stock Price Prediction', fontsize=14, color='#00796b')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.legend()
        
        # Clear previous graph
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        # Display new graph
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
if __name__ == "__main__":
    app = StockMarketSimulatorGUI()
    app.mainloop()
