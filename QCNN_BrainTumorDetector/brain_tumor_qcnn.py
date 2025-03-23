import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import glob
import os
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Configuration
n_qubits = 4  # Number of qubits for quantum circuit
n_layers = 2  # Number of random layers in quantum circuit
image_size = (128, 128)  # Original image size
q_image_size = (32, 32)  # Reduced size for quantum processing
batch_size = 16
n_epochs = 30
device_name = 'default.qubit'  # PennyLane quantum device

class BrainMRIDataset:
    """Class to manage Brain MRI dataset for quantum processing"""
    
    def __init__(self, data_dir="../data/brain_tumor_dataset"):
        self.data_dir = data_dir
        self.tumor_dir = os.path.join(data_dir, "yes")
        self.healthy_dir = os.path.join(data_dir, "no")
        self.images = []
        self.labels = []
        self.load_data()
        
    def load_data(self):
        """Load images and labels from the dataset directories"""
        # Load tumor images
        tumor_images = []
        for f in glob.iglob(os.path.join(self.tumor_dir, "*.jpg")):
            img = cv2.imread(f)
            if img is None:
                print(f"Warning: Could not read image {f}")
                continue
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            tumor_images.append(img)
        
        # Load healthy images
        healthy_images = []
        for f in glob.iglob(os.path.join(self.healthy_dir, "*.jpg")):
            img = cv2.imread(f)
            if img is None:
                print(f"Warning: Could not read image {f}")
                continue
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            healthy_images.append(img)
        
        print(f"Loaded {len(tumor_images)} tumor images and {len(healthy_images)} healthy images")
        
        # Prepare labels
        tumor_labels = np.ones(len(tumor_images))
        healthy_labels = np.zeros(len(healthy_images))
        
        # Combine and convert to numpy arrays
        self.images = np.array(tumor_images + healthy_images, dtype=np.float32)
        self.labels = np.concatenate((tumor_labels, healthy_labels))
        
        # Normalize images to [0, 1]
        self.images = self.images / 255.0
        
        # Shuffle the dataset
        indices = np.arange(len(self.images))
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.labels = self.labels[indices]
        
    def preprocess_for_quantum(self):
        """Preprocess images for quantum processing by downsampling"""
        print("Preprocessing images for quantum computing...")
        downsampled_images = []
        
        for img in self.images:
            # Convert to grayscale to reduce dimensionality
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Resize to smaller dimensions for quantum processing
            small = cv2.resize(gray, q_image_size)
            # Normalize again to [0, 1]
            small = small / 255.0
            downsampled_images.append(small)
        
        return np.array(downsampled_images, dtype=np.float32)
    
    def plot_samples(self, num_samples=5):
        """Plot random samples from each class"""
        tumor_indices = [i for i, label in enumerate(self.labels) if label == 1]
        healthy_indices = [i for i, label in enumerate(self.labels) if label == 0]
        
        tumor_samples = np.random.choice(tumor_indices, min(num_samples, len(tumor_indices)), replace=False)
        healthy_samples = np.random.choice(healthy_indices, min(num_samples, len(healthy_indices)), replace=False)
        
        # Plot tumor samples
        plt.figure(figsize=(15, 3))
        for i, idx in enumerate(tumor_samples):
            plt.subplot(1, num_samples, i+1)
            plt.imshow(self.images[idx])
            plt.title('Tumor')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Plot healthy samples
        plt.figure(figsize=(15, 3))
        for i, idx in enumerate(healthy_samples):
            plt.subplot(1, num_samples, i+1)
            plt.imshow(self.images[idx])
            plt.title('Healthy')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

def create_quantum_circuit(n_qubits, n_layers):
    """Create a quantum circuit with random layers for the quantum convolution"""
    # Use a consistent random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Create a quantum device
    dev = qml.device(device_name, wires=n_qubits)
    
    # Random circuit parameters
    rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, n_qubits))
    
    @qml.qnode(dev)
    def circuit(phi):
        # Encode classical input values
        for j in range(n_qubits):
            qml.RY(np.pi * phi[j], wires=j)
        
        # Apply random quantum circuit
        RandomLayers(rand_params, wires=list(range(n_qubits)))
        
        # Measure in the Z basis
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
    
    return circuit

def quantum_convolution(image, circuit, step=2):
    """
    Apply quantum convolution to the input image.
    
    Args:
        image: Input image of shape (height, width)
        circuit: Quantum circuit function
        step: Step size for moving the convolution window
    
    Returns:
        Quantum convolved image
    """
    h, w = image.shape
    q_h, q_w = h // step, w // step
    out = np.zeros((q_h, q_w, n_qubits))
    
    # Apply quantum circuit to 2x2 patches of the image
    for j in range(0, h - step + 1, step):
        for k in range(0, w - step + 1, step):
            # Process a squared region of the image with a quantum circuit
            patch = [
                image[j, k],
                image[j, k + 1],
                image[j + 1, k],
                image[j + 1, k + 1]
            ]
            
            # Apply quantum circuit and get expectation values
            q_results = circuit(patch)
            
            # Assign expectation values to different channels of the output pixel
            for c in range(n_qubits):
                out[j // step, k // step, c] = q_results[c]
    
    return out

def process_dataset(images, circuit):
    """
    Process the entire dataset using quantum convolution.
    
    Args:
        images: Array of images (num_samples, height, width)
        circuit: Quantum circuit for convolution
    
    Returns:
        Quantum processed images
    """
    q_images = []
    print("Applying quantum convolution to images...")
    for i, img in enumerate(images):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(images)}")
        q_img = quantum_convolution(img, circuit)
        q_images.append(q_img)
    
    return np.array(q_images)

def create_hybrid_model(input_shape):
    """
    Create a hybrid quantum-classical model.
    The first part is quantum (already applied to the images),
    and the second part is a classical neural network.
    
    Args:
        input_shape: Shape of the quantum processed images
    
    Returns:
        Keras model
    """
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_test_split(images, labels, test_size=0.2):
    """Split the dataset into training and testing sets"""
    n_samples = len(images)
    n_test = int(n_samples * test_size)
    
    # Use the first n_test samples for testing and the rest for training
    test_images = images[:n_test]
    test_labels = labels[:n_test]
    train_images = images[n_test:]
    train_labels = labels[n_test:]
    
    return train_images, train_labels, test_images, test_labels

def plot_training_history(history):
    """Plot training history of the model"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix for model evaluation"""
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    cm = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.xticks([0.5, 1.5], ['Healthy', 'Tumor'])
    plt.yticks([0.5, 1.5], ['Healthy', 'Tumor'])
    plt.show()
    
    acc = accuracy_score(y_true, y_pred_binary)
    print(f"Test Accuracy: {acc:.4f}")

def plot_quantum_features(images, q_images, num_samples=3):
    """
    Plot original images and their quantum processed versions
    for a few samples from the dataset.
    """
    indices = np.random.choice(range(len(images)), num_samples, replace=False)
    
    plt.figure(figsize=(12, 4 * num_samples))
    for i, idx in enumerate(indices):
        # Original image
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f"Original Image {idx}")
        plt.axis('off')
        
        # Quantum processed features
        plt.subplot(num_samples, 2, 2*i + 2)
        # Display the first channel of the quantum processed image
        plt.imshow(q_images[idx, :, :, 0], cmap='viridis')
        plt.title(f"Quantum Features (Channel 0)")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Plot all quantum channels for one example
    plt.figure(figsize=(12, 3))
    for c in range(n_qubits):
        plt.subplot(1, n_qubits, c+1)
        plt.imshow(q_images[indices[0], :, :, c], cmap='viridis')
        plt.title(f"Channel {c}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the brain tumor detection using QCNN"""
    # Load and preprocess data
    dataset = BrainMRIDataset()
    dataset.plot_samples()
    
    # Preprocess images for quantum computing
    downsampled_images = dataset.preprocess_for_quantum()
    
    # Create quantum circuit
    circuit = create_quantum_circuit(n_qubits, n_layers)
    
    # Process images with quantum convolution
    q_images = process_dataset(downsampled_images, circuit)
    
    # Display some quantum processed images
    plot_quantum_features(downsampled_images, q_images)
    
    # Split into train and test sets
    train_images, train_labels, test_images, test_labels = train_test_split(q_images, dataset.labels)
    
    # Create and train hybrid model
    input_shape = q_images.shape[1:]  # (height, width, channels)
    model = create_hybrid_model(input_shape)
    
    print(f"Training hybrid quantum-classical model for {n_epochs} epochs...")
    history = model.fit(
        train_images,
        train_labels,
        batch_size=batch_size,
        epochs=n_epochs,
        validation_data=(test_images, test_labels),
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    y_pred = model.predict(test_images)
    plot_confusion_matrix(test_labels, y_pred)
    
    # Save the model
    model.save('QCNN_BrainTumorDetector/model/brain_tumor_qcnn_model')
    print("Model saved successfully!")

if __name__ == "__main__":
    main() 