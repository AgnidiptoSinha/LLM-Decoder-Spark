package test.spark

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.apache.spark.{SparkConf, SparkContext}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet
import spark.{SparkSlidingWindowDataset, WindowData}
import utils.{DataPreprocessor, TokenEmbedding, DimensionValidator, DimensionConfig, InputTokens}
import metrics.{BatchMetrics, GradientStats}
import org.scalatest.BeforeAndAfterAll

class SparkLLMTest extends AnyFlatSpec with Matchers with BeforeAndAfterAll {
  private var sc: SparkContext = _

  override def beforeAll(): Unit = {
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("SparkLLMTest")
      .set("spark.driver.allowMultipleContexts", "true")
    sc = new SparkContext(conf)
  }

  override def afterAll(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "SparkSlidingWindowDataset" should "create correct number of windows" in {
    // Test case 1: Sliding Window Creation
    val tokens = Array("The", "quick", "brown", "fox", "jumps")
    val windowSize = 2
    val embeddingSize = 4

    val slidingWindows = SparkSlidingWindowDataset.createSparkSlidingWindows(
      sc,
      tokens,
      windowSize,
      embeddingSize,
      2 // numPartitions
    )

    // We expect (length - windowSize) windows
    slidingWindows.count() shouldBe (tokens.length - windowSize)

    // Verify the content of windows
    val windowsData = slidingWindows.collect()
    windowsData.foreach { dataset =>
      dataset.getFeatures.shape()(1) shouldBe (windowSize * embeddingSize)
      dataset.getLabels.shape()(1) shouldBe embeddingSize
    }
  }

  "DataPreprocessor" should "handle input size mismatch correctly" in {
    // Test case 2: Data Preprocessing
    val inputSize = 128
    val outputSize = 64
    val preprocessor = DataPreprocessor(inputSize, outputSize)

    // Create test data with incorrect size
    val features = Nd4j.rand(1, inputSize + 10) // Larger than expected
    val labels = Nd4j.rand(1, outputSize - 5)   // Smaller than expected
    val testDataset = new DataSet(features, labels)

    val processedDataset = preprocessor.preprocess(testDataset)

    // Verify dimensions are corrected
    processedDataset.getFeatures.shape()(1) shouldBe inputSize
    processedDataset.getLabels.shape()(1) shouldBe outputSize
  }

  "DimensionValidator" should "validate dimensions correctly" in {
    // Test case 3: Dimension Validation
    val validConfig = DimensionConfig(
      windowSize = 4,
      embeddingSize = 32,
      inputSize = 128, // 4 * 32
      outputSize = 32,
      hiddenSize = 64
    )

    val invalidConfig = validConfig.copy(inputSize = 100) // Invalid input size

    // Test valid configuration
    DimensionValidator.validateDimensions(validConfig) shouldBe true

    // Test invalid configuration
    an [IllegalArgumentException] should be thrownBy {
      DimensionValidator.validateDimensions(invalidConfig)
    }
  }

  "DimensionValidator" should "handle edge cases correctly" in {
    // Test case 4

    // Test large values
    val largeConfig = DimensionConfig(
      windowSize = 1000,
      embeddingSize = 1000,
      inputSize = 1000000,  // 1000 * 1000
      outputSize = 1000,
      hiddenSize = 2000
    )

    DimensionValidator.validateDimensions(largeConfig) shouldBe true

    // Test zero values
    val zeroConfig = DimensionConfig(
      windowSize = 0,
      embeddingSize = 0,
      inputSize = 0,
      outputSize = 0,
      hiddenSize = 0
    )

    an [IllegalArgumentException] should be thrownBy {
      DimensionValidator.validateDimensions(zeroConfig)
    }
  }

  "TokenEmbedding" should "validate embeddings correctly" in {
    // Test case 5: Embedding Validation
    val inputTokens = InputTokens(sc)

    // Create test embeddings
    val validEmbeddings = Vector(
      TokenEmbedding(1, Nd4j.create(Array(0.1, 0.2, 0.3))),
      TokenEmbedding(2, Nd4j.create(Array(0.4, 0.5, 0.6)))
    )

    val invalidEmbeddings = Vector(
      TokenEmbedding(1, Nd4j.create(Array(0.1, 0.2, 0.3))),
      TokenEmbedding(2, Nd4j.create(Array(0.4, 0.5))) // Different size
    )

    // Test validation
    inputTokens.validateEmbeddings(validEmbeddings) shouldBe true
    inputTokens.validateEmbeddings(invalidEmbeddings) shouldBe false

    // Test embedding size
    inputTokens.getEmbeddingSize(validEmbeddings) shouldBe 3
  }
}