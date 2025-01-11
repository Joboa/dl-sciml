import numpy as np
import matplotlib.pyplot as plt

def test_neural_network(network, test_data_path):
    """
    Test the neural network and show results
    """
    test_data = np.loadtxt(test_data_path, delimiter=",")
    
    correct_predictions = 0
    total_samples = len(test_data)
    

    for data in test_data:
        correct_label = int(data[0]) # True label
        inputs = (data[1:] / 255.0 * 0.99) + 0.01
        
        outputs, _ = network.forward(inputs)
        predicted_label = np.argmax(outputs)  # highest probability prediction

        if predicted_label == correct_label:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_samples
    print(f"\nAccuracy: {accuracy:.2%}")
    
    image_predictions(network, test_data, num_examples=5)

def image_predictions(network, test_data, num_examples=5):
    """
    Images and their predictions
    """
    plt.figure(figsize=(2*num_examples, 2))
    
    for i in range(num_examples):
       
        image_data = test_data[i]
        true_label = int(image_data[0])
        
        inputs = (image_data[1:] / 255.0 * 0.99) + 0.01
        outputs, _ = network.forward(inputs)
        predicted_label = np.argmax(outputs)
        
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(image_data[1:].reshape(28, 28), cmap='gray')
        plt.axis('off')
    
        color = 'green' if predicted_label == true_label else 'red'
        plt.title(f'True: {true_label}\nPred: {predicted_label}', color=color)
    
    plt.tight_layout()
    plt.show()

# Test network

test_data_path = "mnist_test_10.csv"
INPUT_NEURONS = 784  # (28x28 pixels)
HIDDEN_NEURONS = 100
OUTPUT_NEURONS = 10  # 10 digits
LEARNING_RATE = 0.01
EPOCHS = 100
    
NN, loss_history = train_network(
            "mnist_test_10.csv",
            INPUT_NEURONS,
            HIDDEN_NEURONS,
            OUTPUT_NEURONS,
            LEARNING_RATE,
            EPOCHS,
            # seed=42
        )
test_neural_network(NN, test_data_path)