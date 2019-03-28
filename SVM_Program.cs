using System;
using System.Collections.Generic;
// for experimentation only. not tested for production use.

namespace SVM_CSharp
{
  class SVM_Program
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin SVM from scratch C# demo\n");

      double[][] train_X = new double[8][] {
        new double[] { 4,5,7 }, new double[] { 7,4,2 },
        new double[] { 0,6,12 }, new double[] { 1,4,8 },
        new double[] { 9,7,5 }, new double[] { 14,7,0 },
        new double[] { 6,9,12 }, new double[] { 8,9,10 }  };

      int[] train_y = new int[8]{ -1, -1, -1, -1, 1, 1, 1, 1 };

      Console.WriteLine("Training data:");
      for (int i = 0; i < train_X.Length; ++i)
      {
        Console.Write("[" + i + "] ");
        for (int j = 0; j < train_X[i].Length; ++j)
          Console.Write(train_X[i][j].ToString("F1").PadLeft(6));
        Console.WriteLine("  |  " + train_y[i].ToString().PadLeft(3));
      }

      Console.WriteLine("\nCreating SVM with poly kernel degree = 2");
      var svm = new SupportVectorMachine("poly", 0); // poly kernel, seed
      svm.gamma = 1.0;
      svm.coef = 0.0;
      svm.degree = 2;

      svm.complexity = 1.0;
      svm.epsilon = 0.001;
      svm.tolerance = 0.001;
      int maxIter = 1000;

      Console.WriteLine("Starting training");
      int iter = svm.Train(train_X, train_y, maxIter);
      Console.WriteLine("Training complete in " + iter + " iterations\n");

      Console.WriteLine("Support vectors:");
      foreach (double[] vec in svm.supportVectors)
      {
        for (int i = 0; i < vec.Length; ++i)
          Console.Write(vec[i].ToString("F1") + "  ");
        Console.WriteLine("");
      }

      Console.WriteLine("\nWeights: ");
      for (int i = 0; i < svm.weights.Length; ++i)
        Console.Write(svm.weights[i].ToString("F6") + " ");
      Console.WriteLine("");

      Console.WriteLine("\nBias = " + svm.bias.ToString("F6") + "\n");

      for (int i = 0; i < train_X.Length; ++i)
      {
        double pred = svm.ComputeDecision(train_X[i]);
        Console.Write("Predicted decision value for [" + i + "] = ");
        Console.WriteLine(pred.ToString("F6").PadLeft(10));
      }

      double acc = svm.Accuracy(train_X, train_y);
      Console.WriteLine("\nModel accuracy on test data = " +
        acc.ToString("F4"));

      double[] unknown = new double[] { 3, 5, 7 };
      double predDecVal = svm.ComputeDecision(unknown);
      Console.WriteLine("\nPredicted value for (3.0 5.0 7.0) = " +
        predDecVal.ToString("F3"));
      int predLabel = Math.Sign(predDecVal);
      Console.WriteLine("\nPredicted label for (3.0 5.0 7.0) = " +
        predLabel);

      Console.WriteLine("\nEnd demo ");
      Console.ReadLine();
    } // Main
  } // Program

  // ======================================================================

  public class SupportVectorMachine
  {
    public Random rnd;

    public double complexity = 1.0;
    public double tolerance = 1.0e-3;  // error tolerance
    public double epsilon = 1.0e-3;

    public List<double[]> supportVectors;  // at least 2 of them
    public double[] weights;  // one weight per support vector
    public double[] alpha;    // one alpha per training item
    public double bias;

    public double[] errors;  // cache. one per training item

    public string kernelType = "poly";
    public double gamma = 1.0;
    public double coef = 0.0;
    public int degree = 2;

    public SupportVectorMachine(string kernelType, int seed)
    {
      if (kernelType != "poly")
        throw new Exception("This SVM uses hard-coded polynomial kernel");
      this.rnd = new Random(seed);
      this.supportVectors = new List<double[]>();
      // this.weights allocated after know how many support vecs there are
    } // ctor

    // --------------------------------------------------------------------

    public double PolyKernel(double[] v1, double[] v2)
    {
      double sum = 0.0;
      for (int i = 0; i < v1.Length; ++i)
        sum += v1[i] * v2[i];
      double z = this.gamma * sum + this.coef;
      return Math.Pow(z, this.degree);
    }

    // --------------------------------------------------------------------

    public double RbfKernel(double[] v1, double[] v2)
    {
      double sum = 0.0;
      for (int i = 0; i < v1.Length; ++i)
        sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
      return Math.Exp(-this.gamma * sum);
    }

    // --------------------------------------------------------------------
    public double ComputeDecision(double[] input)
    {
      double sum = 0;
      for (int i = 0; i < this.supportVectors.Count; ++i) 
        sum += this.weights[i] *
          PolyKernel(this.supportVectors[i], input);
      sum += this.bias;
      return sum;
    }

    // --------------------------------------------------------------------

    public int Train(double[][] X_matrix, int[] y_vector, int maxIter)
    {
      int N = X_matrix.Length;
      this.alpha = new double[N];
      this.errors = new double[N];
      int numChanged = 0;
      bool examineAll = true;
      int iter = 0;

      while (iter < maxIter && numChanged > 0 || examineAll == true)
      {
        ++iter;
        numChanged = 0;
        if (examineAll == true)
        {
          // all training examples
          for (int i = 0; i < N; ++i)
            numChanged += ExamineExample(i, X_matrix, y_vector);
        }
        else
        {
          // examples where alpha is not 0 and not C
          for (int i = 0; i < N; ++i)
            if (this.alpha[i] != 0 && this.alpha[i] != this.complexity)
              numChanged += ExamineExample(i, X_matrix, y_vector);
        }

        if (examineAll == true)
          examineAll = false;
        else if (numChanged == 0)
          examineAll = true;
      }

      List<int> indices = new List<int>();  // indices of support vectors
      for (int i = 0; i < N; ++i)
      {
        // Only store vectors with Lagrange multipliers > 0
        if (this.alpha[i] > 0) indices.Add(i);
      }

      int num_supp_vectors = indices.Count;
      this.weights = new double[num_supp_vectors];
      for (int i = 0; i < num_supp_vectors; ++i)
      {
        int j = indices[i];
        this.supportVectors.Add(X_matrix[j]);
        this.weights[i] = this.alpha[j] * y_vector[j];
      }
      this.bias = -1 * this.bias;
      return iter;
    } // Train

    // --------------------------------------------------------------------
    public double Accuracy(double[][] X_matrix, int[] y_vector)
    {
      // Compute classification accuracy
      int numCorrect = 0; int numWrong = 0;
      for (int i = 0; i < X_matrix.Length; ++i)
      {
        double signComputed = Math.Sign(ComputeDecision(X_matrix[i]));
        if (signComputed == Math.Sign(y_vector[i]))
          ++numCorrect;
        else
          ++numWrong;
      }

      return (1.0 * numCorrect) / (numCorrect + numWrong);
    }

    // --------------------------------------------------------------------

    private bool TakeStep(int i1, int i2,
      double[][] X_matrix, int[] y_vector)
    {
      // "Sequential Minimal Optimization: A Fast Algorithm for
      // Training Support Vector Machines", J. Platt, 1998.
      if (i1 == i2) return false;

      double C = this.complexity;
      double eps = this.epsilon;

      double[] x1 = X_matrix[i1];  // "point" at index i1
      double alph1 = this.alpha[i1];    // Lagrange multiplier for i1
      double y1 = y_vector[i1];    // label

      double E1;
      if (alph1 > 0 && alph1 < C)
        E1 = this.errors[i1];
      else
        E1 = ComputeAll(x1, X_matrix, y_vector) - y1;

      double[] x2 = X_matrix[i2];  // index i2
      double alph2 = this.alpha[i2];
      double y2 = y_vector[i2];

      // SVM output on point [i2] - y2 (check in error cache)
      double E2;
      if (alph2 > 0 && alph2 < C)
        E2 = this.errors[i2];
      else
        E2 = ComputeAll(x2, X_matrix, y_vector) - y2;

      double s = y1 * y2;

      // Compute L and H via equations (13) and (14)
      double L; double H;
      if (y1 != y2)
      {
        L = Math.Max(0, alph2 - alph1);  // 13a
        H = Math.Min(C, C + alph2 - alph1);  // 13b
      }
      else
      {
        L = Math.Max(0, alph2 + alph1 - C);  // 14a
        H = Math.Min(C, alph2 + alph1);  // 14b
      }

      if (L == H) return false;

      double k11 = PolyKernel(x1, x1);  // conveniences
      double k12 = PolyKernel(x1, x2);
      double k22 = PolyKernel(x2, x2);
      double eta = k11 + k22 - 2 * k12;  // 15

      double a1; double a2;
      if (eta > 0)
      {
        a2 = alph2 - y2 * (E2 - E1) / eta;  // 16

        if (a2 >= H) a2 = H;  // 17a
        else if (a2 <= L) a2 = L;  // 17b
      }
      else  // "Under unusual circumstances, eta will not be positive"
      {
        double f1 = 
          y1 * (E1 + this.bias) - alph1 * k11 - s * alph2 * k12;  // 19a
        double f2 =
          y2 * (E2 + this.bias) - alph2 * k22 - s * alph1 * k12;  // 19b
        double L1 = alph1 + s * (alph2 - L);  // 19c
        double H1 = alph1 + s * (alph2 - H);  // 19d
        double Lobj = (L1 * f1) + (L * f2) + (0.5 * L1 * L1 * k11) +
          (0.5 * L * L * k22) + (s * L * L1 * k12);  // 19e
        double Hobj = (H1 * f1) + (H * f2) + (0.5 * H1 * H1 * k11) +
          (0.5 * H * H * k22) + (s * H * H1 * k12);  // 19f

        if (Lobj < Hobj - eps) a2 = L;
        else if (Lobj > Hobj + eps) a2 = H;
        else a2 = alph2;
      }

      if (Math.Abs(a2 - alph2) < eps * (a2 + alph2 + eps))
        return false;

      a1 = alph1 + s * (alph2 - a2);  // 18

      // Update threshold (biasa). See section 2.3
      double b1 = E1 + y1 * (a1 - alph1) * k11 +
        y2 * (a2 - alph2) * k12 + this.bias;
      double b2 = E2 + y1 * (a1 - alph1) * k12 +
        y2 * (a2 - alph2) * k22 + this.bias;
      double newb;
      if (0 < a1 && C > a1)
        newb = b1;
      else if (0 < a2 && C > a2)
        newb = b2;
      else
        newb = (b1 + b2) / 2;

      double deltab = newb - this.bias;
      this.bias = newb;

      // Update error cache using new Lagrange multipliers
      double delta1 = y1 * (a1 - alph1);
      double delta2 = y2 * (a2 - alph2);

      for (int i = 0; i < X_matrix.Length; ++i)
      {
        if (0 < this.alpha[i] && this.alpha[i] < C)
          this.errors[i] += delta1 *
            PolyKernel(x1, X_matrix[i]) +
            delta2 * PolyKernel(x2, X_matrix[i]) - deltab;
      }

      this.errors[i1] = 0.0;
      this.errors[i2] = 0.0;
      this.alpha[i1] = a1;  // Store a1 in the alpha array
      this.alpha[i2] = a2;  // Store a2 in the alpha array

      return true;
    } // TakeStep

    // --------------------------------------------------------------------

    private int ExamineExample(int i2, double[][] X_matrix, int[] y_vector)
    {
      // "Sequential Minimal Optimization: A Fast Algorithm for
      // Training Support Vector Machines", Platt, 1998.
      double C = this.complexity;
      double tol = this.tolerance;

      double[] x2 = X_matrix[i2]; // "point" at i2
      double y2 = y_vector[i2];   // class label for p2
      double alph2 = this.alpha[i2];   // Lagrange multiplier for i2

      // SVM output on point[i2] - y2. (check in error cache)
      double E2;
      if (alph2 > 0 && alph2 < C)
        E2 = this.errors[i2];
      else
        E2 = ComputeAll(x2, X_matrix, y_vector) - y2;

      double r2 = y2 * E2;

      if ((r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0))
      {
        // See section 2.2
        int i1 = -1; double maxErr = 0;
        for (int i = 0; i < X_matrix.Length; ++i)
        {
          if (this.alpha[i] > 0 && this.alpha[i] < C)
          {
            double E1 = this.errors[i];
            double delErr = System.Math.Abs(E2 - E1);

            if (delErr > maxErr)
            {
              maxErr = delErr;
              i1 = i;
            }
          }
        }

        if (i1 >= 0 && TakeStep(i1, i2, X_matrix, y_vector)) return 1;

        int rndi = this.rnd.Next(X_matrix.Length);
        for (i1 = rndi; i1 < X_matrix.Length; ++i1)
        {
          if (this.alpha[i1] > 0 && this.alpha[i1] < C)
            if (TakeStep(i1, i2, X_matrix, y_vector)) return 1;
        }
        for (i1 = 0; i1 < rndi; ++i1)
        {
          if (this.alpha[i1] > 0 && this.alpha[i1] < C)
            if (TakeStep(i1, i2, X_matrix, y_vector)) return 1;
        }

        // "Both the iteration through the non-bound examples and the
        // iteration through the entire training set are started at
        // random locations"
        rndi = this.rnd.Next(X_matrix.Length);
        for (i1 = rndi; i1 < X_matrix.Length; ++i1)
        {
          if (TakeStep(i1, i2, X_matrix, y_vector)) return 1;
        }
        for (i1 = 0; i1 < rndi; ++i1)
        {
          if (TakeStep(i1, i2, X_matrix, y_vector)) return 1;
        }
      } // if ((r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0))

      // "In extremely degenerate circumstances, none of the examples
      // will make an adequate second example. When this happens, the
      // first example is skipped and SMO continues with another chosen
      // first example."
      return 0;
    } // ExamineExample

    // --------------------------------------------------------------------

    private double ComputeAll(double[] vector,
      double[][] X_matrix, int[] y_vector)
    {
      // output using all training data, even if alpha[] is zero
      double sum = -this.bias;  // quirk of SMO paper
      for (int i = 0; i < X_matrix.Length; ++i)
      {
        if (this.alpha[i] > 0)
          sum += this.alpha[i] * y_vector[i] *
            PolyKernel(X_matrix[i], vector);
      }
      return sum;
    }

  } // class SupportVectorMachine
} // ns
// ========================================================================