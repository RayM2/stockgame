import tkinter as tk
from tkcalendar import Calendar
from tkinter import messagebox
from tkinter import ttk
from stock_data import fetch_stock_data
from model import StockPerformanceModel
from utils import preprocess_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta

class StockMarketSimulatorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stock Market Simulator")
        self.geometry("900x700")
        self.balance = 1_000_000  # Starting balance
        self.portfolio = {}
        self.configure_styles()
        self.create_widgets()

    def create_rounded_box(self, canvas, x1, y1, x2, y2, radius, color):
        canvas.create_arc(x1, y1, x1 + radius * 2, y1 + radius * 2, start=90, extent=90, fill=color, outline=color)
        canvas.create_arc(x2 - radius * 2, y1, x2, y1 + radius * 2, start=0, extent=90, fill=color, outline=color)
        canvas.create_arc(x1, y2 - radius * 2, x1 + radius * 2, y2, start=180, extent=90, fill=color, outline=color)
        canvas.create_arc(x2 - radius * 2, y2 - radius * 2, x2, y2, start=270, extent=90, fill=color, outline=color)
        canvas.create_rectangle(x1 + radius, y1, x2 - radius, y2, fill=color, outline=color)
        canvas.create_rectangle(x1, y1 + radius, x2, y2 - radius, fill=color, outline=color)
    
    def configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")  # Use a theme that supports full customization
        
        # General styles for ttk widgets
        style.configure("TFrame", background="#004d40")
        style.configure("TLabel", background="#004d40", foreground="white", font=("Bahnschrift", 12))
        style.configure("TButton", background="#004d40", foreground="white", font=("Bahnschrift", 12))
        style.configure("TEntry", font=("Bahnschrift", 12))      

        # Set the background of the main window
        self.configure(background="#004d40")

    def show_calendar(self, date_type):
        """Display a calendar widget to select a date."""
        # Create a new top-level window for the calendar
        top = tk.Toplevel(self)
        top.title("Select Date")

        # Create the calendar widget
        cal = Calendar(top, selectmode="day", date_pattern="yyyy-mm-dd")
        cal.pack(pady=20)

        # Function to handle date selection
        def on_date_selected():
            selected_date = cal.get_date()
            if date_type == "start":
                self.start_date_entry.delete(0, tk.END)
                self.start_date_entry.insert(0, selected_date)
            elif date_type == "end":
                self.end_date_entry.delete(0, tk.END)
                self.end_date_entry.insert(0, selected_date)
            top.destroy()  # Close the calendar window after selection

        # Add an OK button to confirm the selection
        ok_button = ttk.Button(top, text="OK", command=on_date_selected)
        ok_button.pack(pady=10)
    
    def create_widgets(self):
        # Header Frame for Welcome Text
        self.header_frame = ttk.Frame(self)
        self.header_frame.pack(pady=20)  # Adjust padding to improve spacing

        # Remove rounded box and simplify the canvas
        canvas = tk.Canvas(self.header_frame, width=880, height=80, bg="#004d40", highlightthickness=0)
        canvas.pack()
        canvas.create_text(
            440, 40, 
            text="stock market simulator", 
            font=("Bahnschrift", 20), 
            fill="white"  # Change text color to white
        )

        # Balance Box
        self.balance_frame = ttk.Frame(self)
        self.balance_frame.pack(pady=10)


        self.balance_label = ttk.Label(
            self.balance_frame, 
            text=f"balance: ${self.balance}", 
            font=("Bahnschrift", 16), 
            background="#004d40",  # Match canvas background
            foreground="white"  # Set text color to white
        )
        self.balance_label.pack(pady=10)

        # Input Frame for Stock Ticker and Investment Amount
        self.input_frame = ttk.Frame(self)
        self.input_frame.pack(pady=10)

        self.ticker_label = ttk.Label(self.input_frame, text="stock ticker(eg: AAPL):")
        self.ticker_label.grid(row=0, column=0, padx=5)

        self.ticker_entry = ttk.Entry(self.input_frame)
        self.ticker_entry.grid(row=0, column=1, padx=5)

        self.amount_label = ttk.Label(self.input_frame, text="amount to invest:")
        self.amount_label.grid(row=1, column=0, padx=5)

        self.amount_entry = ttk.Entry(self.input_frame)
        self.amount_entry.grid(row=1, column=1, padx=5)

        self.buy_button = ttk.Button(self.input_frame, text="Buy", command=self.buy_stock)
        self.buy_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Date Frame for End Date Selection
        self.date_frame = ttk.Frame(self)
        self.date_frame.pack(pady=10)

        self.end_date_label = ttk.Label(self.date_frame, text="end date (YYYY-MM-DD):")
        self.end_date_label.grid(row=1, column=0, padx=5)

        self.end_date_entry = ttk.Entry(self.date_frame)
        self.end_date_entry.grid(row=1, column=1, padx=5)

        self.end_date_button = ttk.Button(self.date_frame, text="select date", command=lambda: self.show_calendar("end"))
        self.end_date_button.grid(row=1, column=2, padx=5)

        self.simulate_button = ttk.Button(self.date_frame, text="simulate", command=self.simulate_performance)
        self.simulate_button.grid(row=2, column=0, columnspan=3, pady=10)

        # Portfolio Frame
        self.portfolio_frame = ttk.Frame(self)
        self.portfolio_frame.pack(pady=10)

        self.portfolio_label = ttk.Label(self.portfolio_frame, text="portfolio:", font=("Bahnschrift", 16))
        self.portfolio_label.pack()

        self.portfolio_text = tk.Text(self.portfolio_frame, height=10, width=60, font=("Bahnschrift"))
        self.portfolio_text.pack()

        # Result Frame for Simulation Results
        self.result_frame = ttk.Frame(self)
        self.result_frame.pack(pady=10)

        self.result_label = ttk.Label(self.result_frame, text="simulation results:", font=("Bahnschrift", 16))
        self.result_label.pack()

        self.result_text = tk.Text(self.result_frame, height=10, width=60, font=("Bahnschrift", 12))
        self.result_text.pack()

        # Graph Frame for Plots
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
    
    def simulate_performance(self, step=5):
        try:
            if not self.portfolio:
                messagebox.showerror("Error", "No stocks in portfolio to simulate.")
                return

            end_date = self.end_date_entry.get()
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=None)

            for ticker, amount in self.portfolio.items():
                # Fetch historical data
                training_data = fetch_stock_data(ticker, look_back=250)

                if training_data.empty:
                    messagebox.showerror("Error", f"Failed to fetch data for {ticker}. Skipping.")
                    continue

                if len(training_data) < 61:  # Ensure sufficient data
                    messagebox.showerror("Error", f"Not enough data for {ticker}. Skipping.")
                    continue

                # Train the model
                model = StockPerformanceModel()
                model.train(training_data)

                # Predict future prices iteratively
                current_data = training_data.copy()
                predictions = []
                next_date = current_data.index[-1].to_pydatetime().replace(tzinfo=None) + timedelta(days=1)

                while next_date <= end_date_dt:
                    # Use the last 60 days of data for prediction
                    last_60_days = current_data[-60:]

                    # Ensure exactly 60 rows
                    if len(last_60_days) != 60:
                        messagebox.showerror("Error", f"Not enough data to predict for {ticker}. Skipping further predictions.")
                        break

                    # Extract 'Close' values and flatten
                    last_60_days_values = last_60_days['Close'].values.flatten().reshape(1, -1)

                    # Predict the price for the next day
                    predicted_price = model.predict(last_60_days_values)
                    predictions.append(predicted_price)

                    # Append the predicted price to the dataset
                    current_data = pd.concat(
                        [current_data, pd.DataFrame({'Close': [predicted_price]}, index=[next_date])]
                    )
                    next_date += timedelta(days=1)

                # Calculate results
                final_predicted_price = predictions[-1]
                initial_price = training_data['Close'].iloc[-1]
                num_shares = amount / initial_price
                net_return = (final_predicted_price - initial_price) * num_shares
                percent_return = (net_return / amount) * 100

                # Display results
                result_str = (
                    f"Stock: {ticker}\n"
                    f"Initial Price: ${initial_price:.2f}\n"
                    f"Predicted Price on {end_date}: ${final_predicted_price:.2f}\n"
                    f"Initial Investment: ${amount}\n"
                    f"Net Return: ${net_return:.2f}\n"
                    f"Percent Return: {percent_return:.2f}%\n\n"
                )
                self.result_text.insert(tk.END, result_str)

                # Plot the graph
                self.plot_graph(training_data, final_predicted_price)

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
