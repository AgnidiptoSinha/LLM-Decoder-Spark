ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "2.12.18"

val sparkVersion = "3.5.2"
val dl4jVersion = "1.0.0-beta7"
val nd4jVersion = "1.0.0-beta7"
val slf4jVersion = "2.0.13"
val commonsVersion = "3.15.0"

// Dependency configurations
libraryDependencySchemes ++= Seq(
  "org.scala-lang.modules" %% "scala-xml" % VersionScheme.Always,
  "org.scala-lang.modules" %% "scala-parser-combinators" % VersionScheme.Always
)

// Force specific versions for conflicting dependencies
dependencyOverrides ++= Seq(
  "org.scala-lang.modules" %% "scala-xml" % "2.1.0",
  "org.scala-lang.modules" %% "scala-parser-combinators" % "2.3.0"
)

lazy val root = (project in file("."))
  .settings(
    name := "LLM-Decoder-Spark",
    libraryDependencies ++= Seq(
      // Spark dependencies
      "org.apache.spark" %% "spark-core" % sparkVersion ,
      "org.apache.spark" %% "spark-sql" % sparkVersion ,
      "org.apache.spark" %% "spark-mllib" % sparkVersion ,

      "org.nd4j" % "nd4j-native-platform" % nd4jVersion,
      "org.nd4j" % "nd4j-native" % nd4jVersion classifier "" classifier "windows-x86_64",
      "org.nd4j" % "nd4j-native" % nd4jVersion classifier "" classifier "linux-x86_64",
      "org.nd4j" % "nd4j-native" % nd4jVersion classifier "" classifier "macosx-x86_64",
      "org.nd4j" % "nd4j-api" % nd4jVersion,
      "org.nd4j" % "nd4j-kryo_2.12" % nd4jVersion,

      "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion,
      "org.deeplearning4j" % "deeplearning4j-nlp" % dl4jVersion,
      "org.deeplearning4j" % "dl4j-spark_2.12" % dl4jVersion,
      "org.deeplearning4j" % "dl4j-spark-parameterserver_2.12" % dl4jVersion,

      "org.slf4j" % "slf4j-api" % slf4jVersion,
      "org.slf4j" % "slf4j-simple" % slf4jVersion,
      "ch.qos.logback" % "logback-classic" % "1.2.11",

      // Apache Commons dependencies
      "org.apache.commons" % "commons-math3" % "3.6.1",
      "org.apache.commons" % "commons-lang3" % commonsVersion,
      "commons-io" % "commons-io" % "2.16.1",

      // Configuration Dependencies
      "com.typesafe" % "config" % "1.4.2",

    ),

    // Add resolvers
    resolvers ++= Seq(
      "Maven Central" at "https://repo1.maven.org/maven2/",
      "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
      "Sonatype OSS Releases" at "https://oss.sonatype.org/content/repositories/releases"
    ),

    // Compiler options
    scalacOptions ++= Seq(
      "-deprecation",
      "-feature",
      "-language:postfixOps"
    ),

    // Assembly merge strategy
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
      case PathList("META-INF", "services", xs @ _*) => MergeStrategy.concat
      case PathList("META-INF", xs @ _*) => MergeStrategy.discard
      case "reference.conf" => MergeStrategy.concat
      case x => MergeStrategy.first
    }
)

// Add memory settings for assembly
assembly / javaOptions ++= Seq("-Xmx4G")

assembly / assemblyJarName := "LLMSpark.jar" // Name of your jar file

// Increase memory for forked processes
fork := true
javaOptions ++= Seq(
  "-Xms1024M",
  "-Xmx4096M",
  "-XX:MaxMetaspaceSize=1024M"
)

// Add system property to specify ND4J backend
javaOptions += "-Dorg.nd4j.backend.name=native"
