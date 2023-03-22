package com.databricks.industry.solutions.arc.expressions

import org.apache.spark.sql.catalyst.expressions.{Expression, NullIntolerant, UnaryExpression}
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

case class ARC_EstimateSquaredCountOR(ruleCombination: Expression, ruleSquareCountMap: Map[String, (Long, Long)])
    extends UnaryExpression
      with NullIntolerant
      with CodegenFallback {

    override def child: Expression = ruleCombination

    override def dataType: DataType = LongType

    override def makeCopy(newArgs: Array[AnyRef]): Expression = copy(newArgs(0).asInstanceOf[Expression])

    override def nullSafeEval(input: Any): Any = {
        val arrayData = input.asInstanceOf[ArrayData]
        val rules = arrayData.toArray[UTF8String](StringType).map(_.toString)
        if (rules.length < 2) {
            ruleSquareCountMap.getOrElse(rules.head, (0L, 0L))._1
        } else {
            val multipliers = rules.map(ruleSquareCountMap)
            multipliers.map(x => multipliers.map(y => x._1 * y._2).sum).sum - multipliers.map(x => x._1 * x._2).sum
        }
    }

    override protected def withNewChildInternal(newChild: Expression): Expression = copy(newChild)

}
