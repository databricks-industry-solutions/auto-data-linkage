package com.databricks.industry.solutions.arc.expressions

import org.apache.spark.sql.catalyst.expressions.{CollectionGenerator, Expression, ExpressionInfo}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

case class ARC_GenerateCombinations(
    nCombinationsExpr: Expression,
    elementsExpr: Expression
) extends CollectionGenerator
      with Serializable
      with CodegenFallback {

    override def position: Boolean = false

    override def inline: Boolean = false

    override def children: Seq[Expression] = Seq(nCombinationsExpr, elementsExpr)

    override def eval(input: InternalRow): TraversableOnce[InternalRow] = {
        val arrayData = elementsExpr.eval(input).asInstanceOf[ArrayData]
        val nComb = nCombinationsExpr.eval(input).asInstanceOf[Int]
        val elements = arrayData
            .toArray[UTF8String](StringType)
            .toSeq
            .asInstanceOf[Seq[UTF8String]]
            .map(_.toString)
        val combinations = elements
            .combinations(nComb)
            .toArray
            .map(combination => ArrayData.toArrayData(combination.map(UTF8String.fromString)))
        combinations.map(row => InternalRow.fromSeq(Seq(row)))
    }

    override def elementSchema: StructType = StructType(Seq(StructField("combination", ArrayType(StringType))))

    override def withNewChildrenInternal(newChildren: IndexedSeq[Expression]): Expression = copy(newChildren(0), newChildren(1))

}

object ARC_GenerateCombinations {

    def registryExpressionInfo(db: Option[String]): ExpressionInfo =
        new ExpressionInfo(
          classOf[ARC_GenerateCombinations].getCanonicalName,
          db.orNull,
          "arc_generate_combinations",
          """
            |    _FUNC_(nCombinations, elements)) - Generates the combinations of the input elements.
            |    The number of combinations is specified by the input nCombinations.
            """.stripMargin,
          "",
          "Examples",
          """
            |    > SELECT _FUNC_(2, array("a", "b", "c"));
            |     ["a","b"]
            |     ["a","c"]
            |     ["b","c"]
            """.stripMargin,
          "generator_funcs",
          "1.0",
          "",
          "built-in"
        )
}
