package project.ml.linear

import org.apache.spark.sql.{DataFrame,SparkSession}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder,TrainValidationSplit}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Vectors

object LinearRegressionSimple {
  def main(args:Array[String]){
    
    //val checkpointPath= args(0)
    
    val spark =SparkSession.builder().appName("LinearRegressionSparkJob").enableHiveSupport().getOrCreate()
    //spark.sparkContext.setCheckpointDir(checkpointPath)
    
    import spark.implicits._
    val data= (0 to 10).map(n => (n.toDouble,n.toDouble)).toDF("1st","2nd")
    val df1= data.select(data("1st").as("label"),data("2nd"))
    
    val assembler= new VectorAssembler().setInputCols(Array("2nd")).setOutputCol("features")
    
    val df2 = assembler.transform(df1).select($"label", $"features")
    val lr= new LinearRegression()
    
    val lrModel=lr.fit(df2)
    
    println(s"Coefficients : ${lrModel.coefficients}  Intercept : ${lrModel.intercept}")
    
    val trainingSummary =lrModel.summary
    println(s"Coefficients : ${trainingSummary.totalIterations}  Intercept : ${trainingSummary.objectiveHistory.toList}")
    
    trainingSummary.residuals.show()
    
    val predictions=lrModel.transform(df2)
    
    predictions.show 
  }
}