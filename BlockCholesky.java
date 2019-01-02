package in.ac.iitkgp.atdc;

import java.io.Serializable;
import java.util.List;

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
import org.jblas.Decompose;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import scala.Tuple2;

public class BlockCholesky implements Serializable {

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

	public BlockMatrix blockCholesky(JavaSparkContext sc, BlockMatrix A, int size, int blockSize) {

		if (size == 1) {
			final Broadcast<Integer> bblockSize = sc.broadcast(blockSize);
			JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> inv_L = A.blocks().toJavaRDD().map(
					new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

						@Override
						public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> arg0)
								throws Exception {
							int blockSize = bblockSize.getValue();
							Tuple2<Tuple2<Object, Object>, Matrix> tuple = arg0;
							Tuple2<Object, Object> tuple2 = arg0._1;
							int rowIndex = tuple2._1$mcI$sp();
							int colIndex = tuple2._1$mcI$sp();
							Matrix matrix = arg0._2;
							DoubleMatrix mat = Decompose
									.cholesky(new DoubleMatrix(blockSize, blockSize, matrix.toArray()));
							DoubleMatrix tranMat = mat.transpose();	
							DoubleMatrix matInv = Solve.pinv(tranMat);
							matrix = Matrices.dense(matrix.numRows(), matrix.numCols(), matInv.toArray());
							tuple = new Tuple2<Tuple2<Object, Object>, Matrix>(new Tuple2(rowIndex, colIndex), matrix);
							return tuple;
						}

					});

			BlockMatrix blockLInv = new BlockMatrix(inv_L.rdd(), blockSize, blockSize);
			return blockLInv;
		} else {

			size = size / 2;
			JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD = BlockCholesky.breakMat(A, sc, size);
			BlockMatrix A11 = BlockCholesky._11(pairRDD, sc, blockSize);
			BlockMatrix A12 = BlockCholesky._12(pairRDD, sc, blockSize);
			BlockMatrix A22 = BlockCholesky._22(pairRDD, sc, blockSize);
			
			BlockMatrix L11Inverse = blockCholesky(sc, A11, size, blockSize);						
			BlockMatrix I = L11Inverse.multiply(A12);
			BlockMatrix II = I.transpose();
			BlockMatrix III = II.multiply(I);
			BlockMatrix IV = A22.subtract(III);
			BlockMatrix L22Inverse = blockCholesky(sc, IV, size, blockSize);			
			BlockMatrix V = L22Inverse.multiply(II);
			BlockMatrix VI = V.multiply(L11Inverse);			
			BlockMatrix VII = BlockCholesky.scalerMul(sc, VI, -1, blockSize);			
			BlockMatrix result = BlockCholesky.reArrange(sc, L11Inverse, VII, L22Inverse, size, blockSize);

			return A11;

		}

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

		BlockCholesky blockCholesky = new BlockCholesky();
		SparkConf conf = new SparkConf();
		JavaSparkContext sc = new JavaSparkContext(conf);
		String fileName = args[0];
		String path = "E:\\PenDrive\\BlockCholesky\\" + fileName + ".csv";
		int size = Integer.parseInt(args[1]);
		int blockSize = Integer.parseInt(args[2]);
		int partitionSize = size/blockSize;
		BlockMatrix mat = blockCholesky.getSquareMatrix(sc, path);
		mat.blocks().cache();
		mat.blocks().count();
		long start = System.currentTimeMillis();
		BlockMatrix LInverse = blockCholesky.blockCholesky(sc, mat, partitionSize, blockSize);		
		LInverse.blocks().count();
		long end = System.currentTimeMillis();
		System.out.println((end-start)/1000+" sec.");
		sc.close();
	}

}
