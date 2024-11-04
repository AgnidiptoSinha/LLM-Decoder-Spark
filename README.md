# LLM Decoder Spark Project

## Video Submission

Link: https://youtu.be/HpdfpAmWG0A

## Overview
This project is part of CS441 (Homework 2) focused on building a Large Language Model (LLM) using distributed computing with Apache Spark and DeepLearning4j. The implementation uses Spark's distributed computing capabilities to train a neural network-based decoder for text generation, building upon embedding vectors computed in a previous phase of the project.

## Table of Contents
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Architecture](#architecture)
- [Implementation Details](#implementation-details)
- [Testing](#testing)
- [Performance Metrics](#performance-metrics)
- [License](#license)

## Project Structure
```
.
├── src/main/scala/
│   ├── hw2/
│   │   └── main.scala                 # Main entry point
│   ├── spark/
│   │   ├── LLMTrainer.scala          # Core training logic
│   │   ├── SparkNeuralTrainer.scala  # Neural network training
│   │   ├── SparkSlidingWindow.scala  # Sliding window implementation
│   ├── utils/
│   │   ├── DataPreprocessor.scala    # Data preprocessing utilities
│   │   ├── DimensionValidator.scala  # Model dimension validation
│   │   ├── InputTokens.scala        # Token handling
│   │   └── NeuralTrainerUtils.scala # Training utilities
│   └── metrics/
│       ├── MetricsOutputHandler.scala    # Metrics output formatting
│       └── TrainingMetricsCollector.scala # Training metrics collection
├── src/test/scala/
│   └── test/
│       └── spark/
│           └── SparkLLMTest.scala    # Unit tests
├── build.sbt                         # Project build configuration
└── README.md                         # This file
```

## Prerequisites
- Java JDK 8-22
- Scala 2.12.18
- Apache Spark 3.5.2
- SBT (Scala Build Tool)
- AWS Account (for EMR deployment)

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd llm-decoder-spark
```

2. Install dependencies using SBT:
```bash
sbt clean compile
```

3. Build the assembly JAR:
```bash
sbt assembly
```

## Configuration
The project uses TypeSafe Config for configuration management. Create an `application.conf` file in `src/main/resources/` with the following structure:

```hocon
spark {
  master = "local[*]"  # Change to your Spark master URL
  driverMemory = "4g"
  executorMemory = "4g"
  numWorkers = 4
  targetPartitionSize = 1000
}

model {
  windowSize = 128
  embeddingSize = 512
  hiddenSize = 1024
  outputDir = "models/"
  metricsOutputDir = "metrics/"
}

training {
  batchSize = 32
  numEpochs = 10
  learningRate = 0.001
  trainSplit = 0.8
  validSplit = 0.2
}

data {
  embeddingsPath = "path/to/your/embeddings.txt"
}
```

## Running the Project

### Local Development
1. Run tests:
```bash
sbt test
```

2. Run the application locally:
```bash
sbt run
```

### AWS EMR Deployment
1. Package the application:
```bash
sbt assembly
```

2. Upload the JAR to S3:
```bash
aws s3 cp target/scala-2.12/LLMSpark-assembly.jar s3://your-bucket/
```

3. Create an EMR cluster with Spark installed

4. Submit the Spark job:
```bash
spark-submit \
  --class hw2.main \
  --master yarn \
  --deploy-mode cluster \
  --executor-memory 4G \
  --num-executors 4 \
  s3://your-bucket/LLMSpark-assembly.jar
```

## Architecture

### Core Components

1. **LLMTrainer**: Orchestrates the overall training process, including:
    - Configuration management
    - Spark context initialization
    - Data loading and preprocessing
    - Model training coordination
    - Metrics collection and output

2. **SparkNeuralTrainer**: Handles the neural network training:
    - Network architecture definition
    - Training loop implementation
    - Gradient computation
    - Parameter updates
    - Validation

3. **SparkSlidingWindow**: Implements sliding window dataset creation:
    - Window generation
    - Embedding incorporation
    - Data partitioning
    - Batch preparation

### Data Flow

1. Load token embeddings from previous phase
2. Create sliding windows for context-based training
3. Distribute data across Spark cluster
4. Train neural network using distributed gradient descent
5. Collect and aggregate metrics
6. Save trained model and performance statistics

## Implementation Details

### Neural Network Architecture
- Input Layer: Concatenated context embeddings
- Hidden Layers: Dense layers with ReLU activation
- Output Layer: Linear projection to embedding space
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam

### Distributed Training
- Parameter Averaging Strategy
- Batch-wise processing
- Gradient aggregation across workers
- Periodic model synchronization

### Performance Optimization
- Kryo serialization for efficient data transfer
- Cached RDDs for frequently accessed data
- Optimized partition sizes
- Memory management configurations

## Testing
The project includes unit tests covering:
- Sliding window creation
- Data preprocessing
- Dimension validation
- Embedding validation
- Token handling

Run tests using:
```bash
sbt test
```

## Performance Metrics
The training process collects and outputs:
- Training loss and accuracy
- Validation metrics
- Gradient statistics
- Memory usage
- Batch processing times
- Spark executor statistics
- Data distribution metrics

## License
This project is an academic assignment for CS441 and should be used accordingly.

