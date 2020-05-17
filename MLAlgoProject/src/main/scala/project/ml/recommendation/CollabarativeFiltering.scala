package project.ml.recommendation

import org.apache.spark.sql.{DataFrame,SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.rdd.RDD
import org.apache.spark.ml

object CollabarativeFiltering {
  case class Rating (userId:Int,itemId:Int,rating:Double)
    def main(args:Array[String]){
    val inputFileName=args(0)
    val intermediateOutputPath=args(1)
    val outputPath=args(2)
    val checkpointPath= args(3)
    
    val spark =SparkSession.builder().appName("CollabarativeFilterSparkJob").enableHiveSupport().getOrCreate()
    spark.sparkContext.setCheckpointDir(checkpointPath)
    val dfLoad=spark.read.format("csv").option("header","false").load(inputFileName)
    val colName=Seq("userId","itemId","rating")
    val df= dfLoad.toDF(colName:_*)
    df.createOrReplaceTempView("input_tbl")
    
    val df_distinct_movies=spark.sqlContext.sql("select distinct itemid from input_tbl")
    df_distinct_movies.createOrReplaceTempView("distinct_movies")
    
    val df_distinct_users=spark.sqlContext.sql("select distinct userId from input_tbl")
    df_distinct_users.createOrReplaceTempView("distinct_users")
    
    val cross_movies_user=df_distinct_users.crossJoin(df_distinct_movies)
    cross_movies_user.createOrReplaceTempView("cross_movies_user_tbl")
    
    val df_full=spark.sqlContext.sql("select a.userid,a.itemid, b.itemid b_itemid ,'0' rating from cross_movies_user_tbl a left outer join input_tbl b on a.userid=b.userid and a.itemid=b.itemid")
    df_full.createOrReplaceTempView("full_tbl")
    
    val df_test= spark.sqlContext.sql("select userid,itemid,rating from full_tbl where b_itemid is null")
    
        
    import spark.implicits._
    val ratingDf= df.rdd.map(row =>{
      val userId=row.getString(0)
      val itemId=row.getString(1)
      val ratings = row.getString(2)
      Rating(userId.toInt,itemId.toInt,ratings.toDouble)
    }).toDF()
    
    val testDF= df_test.rdd.map(row =>{
      val userId=row.getString(0)
      val itemId=row.getString(1)
      val ratings = row.getString(2)
      Rating(userId.toInt,itemId.toInt,ratings.toDouble)
    }).toDF()
    
    val trainDF = ratingDf
    val als= new ALS().setMaxIter(20).setRegParam(0.01).setUserCol("userId").setItemCol("itemId").setRatingCol("rating")
    val model=als.fit(trainDF)
    
    val prediction = model.transform(testDF)
    prediction.createOrReplaceTempView("prediction_tbl")
    prediction.orderBy("userid","prediction").show()
    
    val df_rank= spark.sqlContext.sql("select userid,itemid,prediction ,row_number() over (partition by userid order by prediction desc) rnk from prediction_tbl order by rnk ")
    val df_top_recommendation = df_rank.filter("rnk=1")
    df_top_recommendation.show()
    
    prediction.coalesce(1).write.csv(intermediateOutputPath)
    df_top_recommendation.coalesce(1).write.csv(outputPath)
  }
}