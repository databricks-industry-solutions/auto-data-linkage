package com.databricks.industry.solutions.arc.expressions

import org.apache.spark.sql.catalyst.expressions.{Expression, NullIntolerant, UnaryExpression}
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.types.{DataType, StringType}
import org.apache.spark.unsafe.types.UTF8String

case class ARC_ToSplinkRule(ruleCombination: Expression) extends UnaryExpression with NullIntolerant with CodegenFallback {

    override def child: Expression = ruleCombination

    override def dataType: DataType = StringType

    override def makeCopy(newArgs: Array[AnyRef]): Expression = copy(newArgs(0).asInstanceOf[Expression])

    override def nullSafeEval(input: Any): Any = {
        val arrayData = input.asInstanceOf[ArrayData]
        val array = arrayData.toArray[UTF8String](StringType)
        val rule = array
            .map(rule => rule.toString.split(",").map(column => s"l.$column = r.$column").mkString("(", " AND ", ")"))
            .mkString("", " OR ", "")
        UTF8String.fromString(rule)
    }

    override protected def withNewChildInternal(newChild: Expression): Expression = copy(newChild)

}
