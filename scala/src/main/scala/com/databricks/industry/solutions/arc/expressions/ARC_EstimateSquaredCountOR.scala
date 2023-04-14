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
            ruleSquareCountMap.getOrElse(rules.head, (0L, 0L))._2
        } else {
            val orCombinations = rules.combinations(2)
            orCombinations.map(orCombination => {
                val rule1 = orCombination.head
                val rule2 = orCombination.last
                val rule1Stats = ruleSquareCountMap(rule1) // (count, squaredCount)
                val rule2Stats = ruleSquareCountMap(rule2)
                rule1Stats._1 * rule2Stats._2 + rule2Stats._1 * rule1Stats._2 - rule1Stats._2 * rule2Stats._2
            }).sum
        }
    }

    override protected def withNewChildInternal(newChild: Expression): Expression = copy(newChild)

}
