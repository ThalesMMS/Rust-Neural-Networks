#!/usr/bin/env python3
"""
MNIST Digit Recognizer - Drawing App
Loads a trained model from C and allows drawing digits for recognition
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import struct
import sys
import os


class MNISTModel:
    """Neural network model for MNIST digit recognition"""

    def __init__(self):
        self.hidden_weights = None
        self.hidden_biases = None
        self.output_weights = None
        self.output_biases = None
        self.num_inputs = None
        self.num_hidden = None
        self.num_outputs = None

    def load_model(self, filename):
        """Load model weights from binary file saved by C program"""
        with open(filename, 'rb') as f:
            # Read layer dimensions
            self.num_inputs = struct.unpack('i', f.read(4))[0]
            self.num_hidden = struct.unpack('i', f.read(4))[0]
            self.num_outputs = struct.unpack('i', f.read(4))[0]

            print(f"Loading model: {self.num_inputs} -> {self.num_hidden} -> {self.num_outputs}")

            # Read hidden layer weights (input_size x hidden_size)
            self.hidden_weights = np.zeros((self.num_inputs, self.num_hidden))
            for i in range(self.num_inputs):
                weights_data = f.read(8 * self.num_hidden)
                self.hidden_weights[i, :] = struct.unpack(f'{self.num_hidden}d', weights_data)

            # Read hidden layer biases
            biases_data = f.read(8 * self.num_hidden)
            self.hidden_biases = np.array(struct.unpack(f'{self.num_hidden}d', biases_data))

            # Read output layer weights (hidden_size x output_size)
            self.output_weights = np.zeros((self.num_hidden, self.num_outputs))
            for i in range(self.num_hidden):
                weights_data = f.read(8 * self.num_outputs)
                self.output_weights[i, :] = struct.unpack(f'{self.num_outputs}d', weights_data)

            # Read output layer biases
            biases_data = f.read(8 * self.num_outputs)
            self.output_biases = np.array(struct.unpack(f'{self.num_outputs}d', biases_data))

        print(f"Model loaded successfully!")

    @staticmethod
    def relu(x):
        """ReLU activation function"""
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def predict(self, image):
        """
        Predict digit from image
        image: flattened 784-element array, normalized to [0, 1]
        """
        # Hidden layer
        hidden = np.dot(image, self.hidden_weights) + self.hidden_biases
        hidden = self.relu(hidden)

        # Output layer
        output = np.dot(hidden, self.output_weights) + self.output_biases
        output = self.softmax(output)

        return output


class DigitRecognizerApp:
    """Tkinter GUI for drawing and recognizing digits"""

    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("MNIST Digit Recognizer")

        # Drawing canvas settings
        self.canvas_size = 280  # 28x10 for better visibility
        self.brush_size = 20

        # Create PIL image for drawing (28x28)
        self.image = Image.new('L', (28, 28), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # Setup UI
        self.setup_ui()

        # Mouse tracking
        self.last_x = None
        self.last_y = None

    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title = ttk.Label(main_frame, text="Draw a digit (0-9)", font=('Arial', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=10)

        # Drawing canvas
        self.canvas = tk.Canvas(main_frame, width=self.canvas_size, height=self.canvas_size,
                               bg='black', cursor='cross')
        self.canvas.grid(row=1, column=0, padx=10, pady=10)

        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)

        # Prediction panel
        pred_frame = ttk.Frame(main_frame)
        pred_frame.grid(row=1, column=1, padx=10, pady=10, sticky=(tk.N, tk.S))

        pred_label = ttk.Label(pred_frame, text="Predictions:", font=('Arial', 12, 'bold'))
        pred_label.pack(pady=5)

        # Prediction bars
        self.pred_bars = []
        self.pred_labels = []
        for i in range(10):
            frame = ttk.Frame(pred_frame)
            frame.pack(fill=tk.X, pady=2)

            label = ttk.Label(frame, text=f"{i}:", width=2)
            label.pack(side=tk.LEFT)

            bar_canvas = tk.Canvas(frame, width=150, height=20, bg='white')
            bar_canvas.pack(side=tk.LEFT, padx=5)

            prob_label = ttk.Label(frame, text="0.0%", width=6)
            prob_label.pack(side=tk.LEFT)

            self.pred_bars.append(bar_canvas)
            self.pred_labels.append(prob_label)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_canvas)
        clear_btn.pack(side=tk.LEFT, padx=5)

        predict_btn = ttk.Button(button_frame, text="Predict", command=self.predict)
        predict_btn.pack(side=tk.LEFT, padx=5)

        # Instructions
        instr = ttk.Label(main_frame, text="Draw with mouse, then click 'Predict' or it will auto-predict",
                         font=('Arial', 9), foreground='gray')
        instr.grid(row=3, column=0, columnspan=2)

    def on_mouse_down(self, event):
        """Handle mouse button press"""
        self.last_x = event.x
        self.last_y = event.y

    def on_mouse_move(self, event):
        """Handle mouse movement while drawing"""
        if self.last_x and self.last_y:
            # Draw on canvas
            x1, y1 = self.last_x - self.brush_size // 2, self.last_y - self.brush_size // 2
            x2, y2 = self.last_x + self.brush_size // 2, self.last_y + self.brush_size // 2

            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                   fill='white', width=self.brush_size, capstyle=tk.ROUND)

            # Draw on PIL image (scale down to 28x28)
            scale = 28 / self.canvas_size
            x1_img = int(self.last_x * scale)
            y1_img = int(self.last_y * scale)
            x2_img = int(event.x * scale)
            y2_img = int(event.y * scale)

            brush_size_img = max(1, int(self.brush_size * scale))
            self.draw.line([x1_img, y1_img, x2_img, y2_img], fill=255, width=brush_size_img)

            self.last_x = event.x
            self.last_y = event.y

    def on_mouse_up(self, event):
        """Handle mouse button release"""
        self.last_x = None
        self.last_y = None
        # Auto-predict after drawing
        self.predict()

    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete('all')
        self.image = Image.new('L', (28, 28), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.update_predictions(np.zeros(10))

    def predict(self):
        """Run prediction on current drawing"""
        # Convert image to numpy array and normalize
        img_array = np.array(self.image).astype(np.float32) / 255.0
        img_flat = img_array.flatten()

        # Get predictions
        predictions = self.model.predict(img_flat)

        # Update UI
        self.update_predictions(predictions)

    def update_predictions(self, predictions):
        """Update the prediction display"""
        for i in range(10):
            # Update bar
            bar_width = int(predictions[i] * 140)
            self.pred_bars[i].delete('all')
            color = '#4CAF50' if i == np.argmax(predictions) else '#2196F3'
            self.pred_bars[i].create_rectangle(0, 0, bar_width, 20, fill=color)

            # Update label
            self.pred_labels[i].config(text=f"{predictions[i]*100:.1f}%")


def main():
    """Main function"""
    # Determine which model file to load
    model_files = {
        'serial': 'mnist_model.bin',
        'cpu': 'mnist_model_cpu.bin',
        'gpu': 'mnist_model_gpu.bin',
        'cuda': 'mnist_model_gpu.bin'  # CUDA uses same file as GPU
    }

    # Check command line arguments
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
        if model_type not in model_files:
            print(f"Usage: {sys.argv[0]} [serial|cpu|gpu|cuda]")
            print("Default: Tries to find any available model")
            return
        model_file = model_files[model_type]
    else:
        # Try to find any available model
        model_file = None
        for mtype, mfile in model_files.items():
            if os.path.exists(mfile):
                model_file = mfile
                print(f"Found {mtype} model: {mfile}")
                break

        if model_file is None:
            print("Error: No trained model found!")
            print("Available models to train:")
            print("  - mnist_model.bin (serial version)")
            print("  - mnist_model_cpu.bin (OpenMP CPU version)")
            print("  - mnist_model_gpu.bin (OpenMP GPU/CUDA version)")
            print("\nPlease train a model first using one of the C/CUDA programs.")
            return

    # Load the model
    model = MNISTModel()
    try:
        model.load_model(model_file)
    except FileNotFoundError:
        print(f"Error: {model_file} not found!")
        print("Please train the model first using the C program.")
        return

    # Create and run the GUI
    root = tk.Tk()
    app = DigitRecognizerApp(root, model)
    root.mainloop()


if __name__ == '__main__':
    main()
