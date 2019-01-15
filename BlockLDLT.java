package in.ac.iitkgp.atdc;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.distributed.BlockMatrix;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;
import org.apache.spark.rdd.RDD;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import scala.Tuple2;

public class BlockLDLT implements Serializable {

	/**
	 * Breaks the matrix of type [[BlockMatrix]] into four equal sized sub-matrices.
	 * Each block of each sub-matrix gets a tag or key and relative index inside
	 * that sub-matrix.
	 * 
	 * @param A    The input matrix of type [[BlockMatrix]].
	 * @param ctx  The JavaSparkContext of the job.
	 * @param size size size The size of the matrix in terms of number of
	 *             partitions. If the dimension of the matrix is n and the dimension
	 *             of each block is m, the value of size is = n/m.
	 * @return PairRDD `pairRDD` of [[<String, BlockMatrix.RDD>]] type. Each tuple
	 *         consists of a tag corresponds to block's coordinate and the RDD of
	 *         blocks.
	 */

	private static JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> breakMat(BlockMatrix A,
			JavaSparkContext ctx, int size) {

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rdd = A.blocks().toJavaRDD();
		final Broadcast<Integer> bSize = ctx.broadcast(size);
		JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD = rdd.mapToPair(
				new PairFunction<Tuple2<Tuple2<Object, Object>, Matrix>, String, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> call(
							Tuple2<Tuple2<Object, Object>, Matrix> tuple) throws Exception {

						Tuple2<Object, Object> tuple1 = tuple._1;
						int rowIndex = tuple1._1$mcI$sp();
						int colIndex = tuple1._2$mcI$sp();
						Matrix matrix = tuple._2;
						String tag = "";
						if (rowIndex / bSize.value() == 0 && colIndex / bSize.value() == 0) {
							rowIndex = rowIndex % bSize.value();
							colIndex = colIndex % bSize.value();
							tag = "A11";
						} else if (rowIndex / bSize.value() == 0 && colIndex / bSize.value() == 1) {
							rowIndex = rowIndex % bSize.value();
							colIndex = colIndex % bSize.value();
							tag = "A12";
						} else if (rowIndex / bSize.value() == 1 && colIndex / bSize.value() == 0) {
							rowIndex = rowIndex % bSize.value();
							colIndex = colIndex % bSize.value();
							tag = "A21";
						} else {
							rowIndex = rowIndex % bSize.value();
							colIndex = colIndex % bSize.value();
							tag = "A22";
						}
						return new Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>(tag,
								new Tuple2(new Tuple2(rowIndex, colIndex), matrix));
					}

				});

		return pairRDD;
	}

	/**
	 * Returns the upper-left sub-matrix of a broken matrix.
	 * 
	 * @param pairRDD   The PairRDD of a broken RDD with a tag for each block.
	 * @param ctx       The JavaSparkContext for the job.
	 * @param blockSize The block size of the matrix.
	 * @return the upper-left sub-matrix of type [[BlockMatrix]] of a broken matrix.
	 */

	private static BlockMatrix _11(JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD,
			JavaSparkContext ctx, int blockSize) {
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rddEntry = pairRDD
				.filter(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Boolean>() {

					@Override
					public Boolean call(Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple2 = tuple;
						String tag = tuple2._1;
						return tag.equals("A11");
					}
				})
				.map(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(
							Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						return tuple._2;
					}

				});
		BlockMatrix matA = new BlockMatrix(rddEntry.rdd(), blockSize, blockSize);
		return matA;
	}

	/**
	 * Returns the upper-right sub-matrix of a broken matrix.
	 * 
	 * @param pairRDD   The PairRDD of a broken RDD with a tag for each block.
	 * @param ctx       The JavaSparkContext for the job.
	 * @param blockSize The block size of the matrix.
	 * @return the upper-left sub-matrix of type [[BlockMatrix]] of a broken matrix.
	 */

	private static BlockMatrix _12(JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD,
			JavaSparkContext ctx, int blockSize) {
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rddEntry = pairRDD
				.filter(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Boolean>() {

					@Override
					public Boolean call(Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple2 = tuple;
						String tag = tuple2._1;
						return tag.equals("A12");
					}
				})
				.map(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(
							Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						return tuple._2;
					}

				});
		BlockMatrix matA = new BlockMatrix(rddEntry.rdd(), blockSize, blockSize);
		return matA;
	}

	/**
	 * Returns the lower-left sub-matrix of a broken matrix.
	 * 
	 * @param pairRDD   The PairRDD of a broken RDD with a tag for each block.
	 * @param ctx       The JavaSparkContext for the job.
	 * @param blockSize The block size of the matrix.
	 * @return the upper-left sub-matrix of type [[BlockMatrix]] of a broken matrix.
	 */

	private static BlockMatrix _21(JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD,
			JavaSparkContext ctx, int blockSize) {
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rddEntry = pairRDD
				.filter(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Boolean>() {

					@Override
					public Boolean call(Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple2 = tuple;
						String tag = tuple2._1;
						return tag.equals("A21");
					}
				})
				.map(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(
							Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						return tuple._2;
					}

				});
		BlockMatrix matA = new BlockMatrix(rddEntry.rdd(), blockSize, blockSize);
		return matA;
	}

	/**
	 * Returns the lower-right sub-matrix of a broken matrix.
	 * 
	 * @param pairRDD   The PairRDD of a broken RDD with a tag for each block.
	 * @param ctx       The JavaSparkContext for the job.
	 * @param blockSize The block size of the matrix.
	 * @return the upper-left sub-matrix of type [[BlockMatrix]] of a broken matrix.
	 */

	private static BlockMatrix _22(JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD,
			JavaSparkContext ctx, int blockSize) {
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rddEntry = pairRDD
				.filter(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Boolean>() {

					@Override
					public Boolean call(Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple2 = tuple;
						String tag = tuple2._1;
						return tag.equals("A22");
					}
				})
				.map(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(
							Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						return tuple._2;
					}

				});
		BlockMatrix matA = new BlockMatrix(rddEntry.rdd(), blockSize, blockSize);
		return matA;
	}

	private static BlockMatrix reArrange(JavaSparkContext ctx, BlockMatrix C11, BlockMatrix C21, BlockMatrix C22,
			int size, int blockSize) {
		final Broadcast<Integer> bSize = ctx.broadcast(size);
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C11_RDD = C11.blocks().toJavaRDD();
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C21_RDD = C21.blocks().toJavaRDD();
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C22_RDD = C22.blocks().toJavaRDD();

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C21Arranged = C21_RDD
				.map(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> tuple)
							throws Exception {
						int size = bSize.getValue();
						int rowIndex = tuple._1._1$mcI$sp() + size;
						int colIndex = tuple._1._2$mcI$sp();
						Matrix matrix = tuple._2;
						return new Tuple2(new Tuple2(rowIndex, colIndex), matrix);
					}
				});

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C22Arranged = C22_RDD
				.map(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> tuple)
							throws Exception {
						int size = bSize.getValue();
						int rowIndex = tuple._1._1$mcI$sp() + size;
						int colIndex = tuple._1._2$mcI$sp() + size;
						Matrix matrix = tuple._2;
						return new Tuple2(new Tuple2(rowIndex, colIndex), matrix);
					}
				});

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> union = C11_RDD.union(C21Arranged.union(C22Arranged));
		BlockMatrix C = new BlockMatrix(union.rdd(), blockSize, blockSize);
		return C;
	}

	private static BlockMatrix reArrange(JavaSparkContext ctx, BlockMatrix C11, BlockMatrix C22, int size,
			int blockSize) {
		final Broadcast<Integer> bSize = ctx.broadcast(size);
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C11_RDD = C11.blocks().toJavaRDD();
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C22_RDD = C22.blocks().toJavaRDD();
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C22Arranged = C22_RDD
				.map(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> tuple)
							throws Exception {
						int size = bSize.getValue();
						int rowIndex = tuple._1._1$mcI$sp() + size;
						int colIndex = tuple._1._2$mcI$sp() + size;
						Matrix matrix = tuple._2;
						return new Tuple2(new Tuple2(rowIndex, colIndex), matrix);
					}
				});

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> union = C11_RDD.union(C22Arranged);
		BlockMatrix C = new BlockMatrix(union.rdd(), blockSize, blockSize);
		return C;
	}

	private static BlockMatrix scalerMul(JavaSparkContext ctx, BlockMatrix A, final double scalar, int blockSize) {
		final Broadcast<Integer> bblockSize = ctx.broadcast(blockSize);
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> A_RDD = A.blocks().toJavaRDD();
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> B_RDD = A_RDD
				.map(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> tuple)
							throws Exception {
						int blockSize = bblockSize.getValue();
						Tuple2<Tuple2<Object, Object>, Matrix> tuple2 = tuple;
						int rowIndex = tuple2._1._1$mcI$sp();
						int colIndex = tuple2._1._2$mcI$sp();
						Matrix matrix = tuple._2;
						DoubleMatrix candidate = new DoubleMatrix(matrix.toArray());
						DoubleMatrix product = candidate.muli(scalar);
						matrix = Matrices.dense(blockSize, blockSize, product.toArray());
						return new Tuple2(new Tuple2(rowIndex, colIndex), matrix);
					}
				});

		BlockMatrix product = new BlockMatrix(B_RDD.rdd(), blockSize, blockSize);
		return product;

	}

	public void print(BlockMatrix blockMat) {

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> blockRDD = blockMat.blocks().toJavaRDD();
		List<Tuple2<Tuple2<Object, Object>, Matrix>> blockList = blockRDD.collect();

		for (int i = 0; i < blockList.size(); i++) {
			System.out.println("[" + blockList.get(i)._1._1$mcI$sp() + ":" + blockList.get(i)._1._2$mcI$sp() + "]");

			int numRows = blockList.get(i)._2.numRows();
			int numCols = blockList.get(i)._2.numCols();
			for (int j = 0; j < numRows; j++) {
				for (int k = 0; k < numCols; k++) {
					System.out.println(blockList.get(i)._2.apply(j, k));
				}
			}

		}
	}

	public double[][] getMatrix(int size, int min, int max) {
		double[][] mat = new double[size][size];
		Random r = new Random();
		for (int i = 0; i < size; i++) {
			for (int j = 0; j <= size; j++) {
				double random = min + r.nextDouble() * (max - min);
				if (i != j) {
					mat[i][j] = random;
					mat[j][i] = random;
				} else {
					mat[i][j] = random;
				}
			}
		}
		return mat;
	}

	public List<Tuple2<BlockMatrix, BlockMatrix>> blockLDLT(JavaSparkContext sc, BlockMatrix A, int size,
			int blockSize) {

		if (size == 1) {
			List<Tuple2<BlockMatrix, BlockMatrix>> listTuple = new ArrayList<Tuple2<BlockMatrix, BlockMatrix>>();
			List<Tuple2<Tuple2<Object, Object>, Matrix>> list = new ArrayList<Tuple2<Tuple2<Object, Object>, Matrix>>();
			list = A.blocks().toJavaRDD().collect();

			int rowIndex = list.get(0)._1._1$mcI$sp();
			int colIndex = list.get(0)._1._2$mcI$sp();
			Matrix matrix = list.get(0)._2;

			double[][] a = new double[matrix.numRows()][matrix.numCols()];

			for (int j = 0; j < matrix.numCols(); j++) {
				for (int i = 0; i < matrix.numRows(); i++) {
					a[i][j] = matrix.apply(i, j);
				}
			}

			Tuple2<double[][], double[][]> LD = serialLDL(a, blockSize);
			double[][] L = LD._1;
			double[][] D = LD._2;
			double[] lArray = new double[blockSize * blockSize];
			double[] dArray = new double[blockSize * blockSize];

			lArray = toSingleArray(L);
			dArray = toSingleArray(D);

			Matrix LMatrix = Matrices.dense(matrix.numRows(), matrix.numCols(), lArray);
			Matrix DMatrix = Matrices.dense(matrix.numRows(), matrix.numCols(), dArray);

			List<Tuple2<Tuple2<Object, Object>, Matrix>> LList = new ArrayList<Tuple2<Tuple2<Object, Object>, Matrix>>();
			List<Tuple2<Tuple2<Object, Object>, Matrix>> DList = new ArrayList<Tuple2<Tuple2<Object, Object>, Matrix>>();
			LList.add(new Tuple2(new Tuple2(rowIndex, colIndex), LMatrix));
			DList.add(new Tuple2(new Tuple2(rowIndex, colIndex), DMatrix));
			RDD<Tuple2<Tuple2<Object, Object>, Matrix>> LRDD = sc.parallelize(LList).rdd();
			RDD<Tuple2<Tuple2<Object, Object>, Matrix>> DRDD = sc.parallelize(DList).rdd();
			BlockMatrix LMat = new BlockMatrix(LRDD, blockSize, blockSize);
			BlockMatrix DMat = new BlockMatrix(DRDD, blockSize, blockSize);

			listTuple.add(new Tuple2(LMat, DMat));

			DoubleMatrix lInverse = Solve.pinv(new DoubleMatrix(blockSize, blockSize, lArray));
			double[] dInverse = diagInverse(dArray);

			Matrix LInvMatrix = Matrices.dense(matrix.numRows(), matrix.numCols(), lInverse.toArray());
			Matrix DInvMatrix = Matrices.dense(matrix.numRows(), matrix.numCols(), dInverse);

			List<Tuple2<Tuple2<Object, Object>, Matrix>> LInvList = new ArrayList<Tuple2<Tuple2<Object, Object>, Matrix>>();
			List<Tuple2<Tuple2<Object, Object>, Matrix>> DInvList = new ArrayList<Tuple2<Tuple2<Object, Object>, Matrix>>();
			LInvList.add(new Tuple2(new Tuple2(rowIndex, colIndex), LInvMatrix));
			DInvList.add(new Tuple2(new Tuple2(rowIndex, colIndex), DInvMatrix));
			RDD<Tuple2<Tuple2<Object, Object>, Matrix>> LInvRDD = sc.parallelize(LInvList).rdd();
			RDD<Tuple2<Tuple2<Object, Object>, Matrix>> DInvRDD = sc.parallelize(DInvList).rdd();
			BlockMatrix LInv = new BlockMatrix(LInvRDD, blockSize, blockSize);
			BlockMatrix DInv = new BlockMatrix(DInvRDD, blockSize, blockSize);

			listTuple.add(new Tuple2(LInv, DInv));

			return listTuple;
		} else {
			List<Tuple2<BlockMatrix, BlockMatrix>> listTuple = new ArrayList<Tuple2<BlockMatrix, BlockMatrix>>();
			size = size / 2;
			JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD = BlockLDLT.breakMat(A, sc, size);
			BlockMatrix A11 = BlockLDLT._11(pairRDD, sc, blockSize);
			BlockMatrix A21 = BlockLDLT._21(pairRDD, sc, blockSize);
			BlockMatrix A22 = BlockLDLT._22(pairRDD, sc, blockSize);

			List<Tuple2<BlockMatrix, BlockMatrix>> ldlt1 = blockLDLT(sc, A11, size, blockSize);
			BlockMatrix D11 = ldlt1.get(0)._2;
			BlockMatrix L11Inv = ldlt1.get(1)._1;
			BlockMatrix D11Inv = ldlt1.get(1)._2;
			BlockMatrix I = L11Inv.transpose();
			BlockMatrix II = A21.multiply(I);
			BlockMatrix III = II.multiply(D11Inv);
			BlockMatrix IV = III.transpose();
			BlockMatrix V = III.multiply(D11);
			BlockMatrix VI = V.multiply(IV);
			BlockMatrix VII = A22.subtract(VI);
			List<Tuple2<BlockMatrix, BlockMatrix>> ldlt2 = blockLDLT(sc, VII, size, blockSize);
			BlockMatrix L22Inv = ldlt2.get(1)._1;
			BlockMatrix D22Inv = ldlt2.get(1)._2;

			BlockMatrix VIII = L22Inv.multiply(III);
			BlockMatrix IX = VIII.multiply(L11Inv);
			BlockMatrix X = scalerMul(sc, IX, -1, blockSize);

			BlockMatrix LInv = reArrange(sc, L11Inv, X, L22Inv, size, blockSize);
			BlockMatrix DInv = reArrange(sc, D11Inv, D22Inv, size, blockSize);
			listTuple.add(new Tuple2(LInv, DInv));
			return listTuple;

		}

	}

	public double[] diagInverse(double[] D) {
		int row = (int) Math.sqrt(D.length);
		double[] inv = new double[D.length];
		for (int i = 0; i < row; i++) {
			inv[i * row + i] = 1 / D[i * row + i];
		}
		return inv;
	}

	public double[] toSingleArray(double[][] arr) {
		double[] sArr = new double[arr.length * arr[0].length];
		int k = 0;
		for (int j = 0; j < arr.length; j++) {
			for (int i = 0; i < arr.length; i++) {
				sArr[k] = arr[i][j];
				k++;
			}
		}

		return sArr;
	}

	public Tuple2<double[][], double[][]> serialLDL(double[][] A, int size) {
		double[][] L = new double[size][size];
		double[][] D = new double[size][size];

		for (int i = 0; i < size; i++) {
			double sum = A[i][i];

			for (int j = 0; j < i; j++) {
				sum = sum - (Math.pow(L[i][j], 2) * D[j][j]);
			}

			D[i][i] = sum;

			for (int j = i; j < size; j++) {
				sum = A[j][i];

				for (int k = 0; k < i; k++) {
					sum = sum - ((L[j][k] * L[i][k]) * D[k][k]);
				}

				L[j][i] = sum / D[i][i];
			}
		}
		return new Tuple2<double[][], double[][]>(L, D);

	}

	public BlockMatrix getSquareMatrix(JavaSparkContext sc, String path) {
		JavaRDD<String> lines = sc.textFile(path);
		JavaRDD<MatrixEntry> mat = lines.map(new Function<String, MatrixEntry>() {

			@Override
			public MatrixEntry call(String line) throws Exception {
				long row = Long.parseLong(line.split(",")[0]);
				long column = Long.parseLong(line.split(",")[1]);
				double value = Double.parseDouble(line.split(",")[2]);

				MatrixEntry entry = new MatrixEntry(row, column, value);
				return entry;
			}
		});
		CoordinateMatrix coordinateMatrix = new CoordinateMatrix(mat.rdd());
		BlockMatrix matrix = coordinateMatrix.toBlockMatrix(2, 2);
		return matrix;
	}

	public static void main(String args[]) {
		List<Tuple2<BlockMatrix, BlockMatrix>> listTuple = new ArrayList<Tuple2<BlockMatrix, BlockMatrix>>();
		BlockLDLT blockLDLT = new BlockLDLT();
		SparkConf conf = new SparkConf();
		JavaSparkContext sc = new JavaSparkContext(conf);

		String fileName = args[0];
		String path = "E:\\PenDrive\\BlockLDLT\\" + fileName + ".csv";
		int size = Integer.parseInt(args[1]);
		int blockSize = Integer.parseInt(args[2]);

		int partitionSize = size / blockSize;
		BlockMatrix mat = blockLDLT.getSquareMatrix(sc, path);
		mat.blocks().cache();
		mat.blocks().count();
		long start = System.currentTimeMillis();
		listTuple = blockLDLT.blockLDLT(sc, mat, partitionSize, blockSize);
		listTuple.get(0)._1.blocks().count();
		long end = System.currentTimeMillis();
		System.out.println((end - start) / 1000 + " sec.");
		sc.close();
	}

}