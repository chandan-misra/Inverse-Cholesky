package in.ac.iitkgp.atdc;

import java.io.Serializable;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.BlockMatrix;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;
import org.jblas.Decompose;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import scala.Tuple2;

public class POTRF implements Serializable{

	public static void main(String args[]) {

		POTRF potrf = new POTRF();
		SparkConf conf = new SparkConf();
		JavaSparkContext sc = new JavaSparkContext(conf);

		// getting a matrix
		String matrix = args[0];
		String matrixPath = "file:///E:/3rd-Paper/" + matrix;

		// getting the block size
		int blockSize = Integer.parseInt(args[1]);		

		// calculating the split size
		int size = (int) sc.textFile(matrixPath).count();
		int splitSize = (int)Math.ceil(size / blockSize);
		System.out.println(splitSize);
		
		BlockMatrix mat = potrf.getSquareMatrix(sc, matrixPath, blockSize);

		potrf.decompose(sc, mat, splitSize, blockSize);
	}

	public void decompose(JavaSparkContext sc, BlockMatrix mat, int splitSize, int blockSize) {

		int i = 0;
		
		while (i < (splitSize - 1)) {
			Broadcast<Integer> index = sc.broadcast(i);
			Broadcast<Integer> bBlockSize = sc.broadcast(blockSize);
			Broadcast<Integer> splitBroadcast = sc.broadcast(splitSize);
			JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rdd = mat.blocks().toJavaRDD();
			
			//print(mat);

			JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> A11 = rdd
					.filter(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Boolean>() {

						@Override
						public Boolean call(Tuple2<Tuple2<Object, Object>, Matrix> tuple) throws Exception {
							Tuple2<Object, Object> tuple1 = tuple._1;
							int rowIndex = tuple1._1$mcI$sp();
							int colIndex = tuple1._2$mcI$sp();
							int i = index.getValue();
							return (rowIndex == i && colIndex == i);
						}
					});
			
			print(new BlockMatrix(A11.rdd(), blockSize, blockSize));

			JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> L11TInvRDD = A11.map(
					new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

						@Override
						public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> tuple)
								throws Exception {
							int blockSize = bBlockSize.getValue();
							int rowIndex = tuple._1._1$mcI$sp();
							int colIndex = tuple._1._2$mcI$sp();
							Matrix matrix = tuple._2;
							DoubleMatrix mat = Decompose
									.cholesky(new DoubleMatrix(blockSize, blockSize, matrix.toArray()));
							//DoubleMatrix tranMat = mat.transpose();
							DoubleMatrix matInv = Solve.pinv(mat);
							matrix = Matrices.dense(matrix.numRows(), matrix.numCols(), matInv.toArray());
							tuple = new Tuple2<Tuple2<Object, Object>, Matrix>(new Tuple2(rowIndex, colIndex), matrix);
							return tuple;
						}

					});

			
			print(new BlockMatrix(L11TInvRDD.rdd(), blockSize, blockSize));
			
			BlockMatrix L11TInv = new BlockMatrix(L11TInvRDD.rdd(), blockSize, blockSize);
			
			JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> A21RDD = rdd.filter(new Function<Tuple2<Tuple2<Object,Object>,Matrix>, Boolean>() {
				
				@Override
				public Boolean call(Tuple2<Tuple2<Object, Object>, Matrix> tuple) throws Exception {
					Tuple2<Object, Object> tuple1 = tuple._1;
					int rowIndex = tuple1._1$mcI$sp();
					int colIndex = tuple1._2$mcI$sp();
					int i = index.getValue();
					return (rowIndex >= (i+1) && colIndex == 0);
				}
			});
			
			print(new BlockMatrix(A21RDD.rdd(), blockSize, blockSize));
			
			BlockMatrix A21 = new BlockMatrix(A21RDD.rdd(), blockSize, blockSize);
			
			BlockMatrix L21 = A21.multiply(L11TInv);
			print(L21);
			
			JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> A22RDD = rdd.filter(new Function<Tuple2<Tuple2<Object,Object>,Matrix>, Boolean>() {
				
				@Override
				public Boolean call(Tuple2<Tuple2<Object, Object>, Matrix> tuple) throws Exception {					
					Tuple2<Object, Object> tuple1 = tuple._1;
					int rowIndex = tuple1._1$mcI$sp();
					int colIndex = tuple1._2$mcI$sp();
					int i = index.getValue();
					return (rowIndex >= (i+1) && colIndex >= (i+1));
				}
			});
			
			BlockMatrix A22 = new BlockMatrix(A22RDD.rdd(), blockSize, blockSize);
			print(A22);
			 
			BlockMatrix L21T = L21.transpose();
			BlockMatrix L21L21T = L21.multiply(L21T);
			mat = A22.subtract(L21L21T);
			print(mat);
			JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> L22 = null;
			if(i == (splitSize - 2)) {
				//single node decomposition
				L22 = mat.blocks().toJavaRDD().map(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> tuple)
							throws Exception {
						int blockSize = bBlockSize.getValue();
						Tuple2<Tuple2<Object, Object>, Matrix> tuple1 = tuple;
						Tuple2<Object, Object> tuple2 = tuple1._1;
						int rowIndex = tuple2._1$mcI$sp();
						int colIndex = tuple2._1$mcI$sp();
						Matrix matrix = tuple._2;
						DoubleMatrix mat = Decompose
								.cholesky(new DoubleMatrix(blockSize, blockSize, matrix.toArray()));
						matrix = Matrices.dense(matrix.numRows(), matrix.numCols(), mat.toArray());
						tuple = new Tuple2<Tuple2<Object, Object>, Matrix>(new Tuple2(rowIndex, colIndex), matrix);
						return tuple;
					}
				});
				
				
			}
			//rearrange
			
			i++;
			
			
		}
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
					System.out.println("["+j+"]["+k+"]: "+blockList.get(i)._2.apply(j, k));
				}
			}

		}
	}

	public BlockMatrix getSquareMatrix(JavaSparkContext sc, String path, int blockSize) {
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
		BlockMatrix matrix = coordinateMatrix.toBlockMatrix(blockSize, blockSize);
		return matrix;
	}

}