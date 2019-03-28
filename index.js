'use strict'

let svm = {}

svm.supportVectors = []

svm.rnd = function getRandomInt (min, max) {
  min = Math.ceil(min)
  max = Math.floor(max)
  return Math.floor(Math.random() * (max - min)) + min // The maximum is exclusive and the minimum is inclusive
}

svm.complexity = 1.0
svm.tolerance = 1.0e-3
svm.epsilon = 1.0e-3

svm.weights = []
svm.alpha = []
svm.bias = 0

svm.errors = []
svm.kernelType = 'poly'
svm.gamma = 1.0
svm.coef = 0.0
svm.degree = 2

svm.complexity = 1.0
svm.epsilon = 0.001
svm.tolerance = 0.001
svm.maxIter = 1000

svm.polyKernel = async function (v1, v2) {
  let sum = 0.0
  for (let i = 0; i < v1.length; i++) {
    sum += v1[i] * v2[i]
  }
  let z = svm.gamma * sum + svm.coef
  return Math.pow(z, svm.degree)
}

svm.rbfKernel = async function (v1, v2) {
  let sum = 0.0
  for (let i = 0; i < v1.length; i++) {
    sum += (v1[i] - v2[i]) * (v1[i] - v2[i])
  }
  return Math.exp(-svm.gamma * sum)
}

svm.computeDecision = async function (input) {
  let sum = 0
  for (let i = 0; i < svm.supportVectors.length; i++) {
    sum += svm.weights[i] * await svm.polyKernel(svm.supportVectors[i], input)
  }
  sum += svm.bias
  return sum
}

svm.accuracy = async function (xMatrix, yVector) {
  let numCorrect = 0
  let numWrong = 0

  for (let i = 0; i < xMatrix.length; i++) {
    let signComputed = Math.sign(await svm.computeDecision(xMatrix[i]))
    if (signComputed === Math.sign(yVector[i])) {
      numCorrect++
    } else {
      numWrong++
    }
  }

  return (1.0 * numCorrect) / (numCorrect + numWrong)
}

svm.computeAll = async function (vector, xMatrix, yVector) {
  // output using all training data, even if alpha[] is zero
  let sum = -svm.bias // quirk of SMO paper
  for (let i = 0; i < xMatrix.length; i++) {
    if (svm.alpha[i] > 0) {
      sum += svm.alpha[i] * yVector[i] * await svm.polyKernel(xMatrix[i], vector)
    }
  }

  return sum
}

svm.takeStep = async function (i1, i2, xMatrix, yVector) {
  // "Sequential Minimal Optimization: A Fast Algorithm for
  // Training Support Vector Machines", J. Platt, 1998.
  if (i1 === i2) return false

  let C = svm.complexity
  let eps = svm.epsilon

  let x1 = xMatrix[i1] // "point" at index i1
  let alph1 = svm.alpha[i1] // Lagrange multiplier for i1
  let y1 = yVector[i1] // label

  let e1
  if (alph1 > 0 && alph1 < C) {
    e1 = svm.errors[i1]
  } else {
    e1 = await svm.computeAll(x1, xMatrix, yVector) - y1
  }

  let x2 = xMatrix[i2] // index i2
  let alph2 = svm.alpha[i2]
  let y2 = yVector[i2]

  // SVM output on point [i2] - y2 (check in error cache)
  let e2
  if (alph2 > 0 && alph2 < C) {
    e2 = svm.errors[i2]
  } else {
    e2 = await svm.computeAll(x2, xMatrix, yVector) - y2
  }

  let s = y1 * y2

  // Compute L and H via equations (13) and (14)
  let L
  let H

  if (y1 !== y2) {
    L = Math.max(0, alph2 - alph1) // 13a
    H = Math.min(C, C + alph2 - alph1) // 13b
  } else {
    L = Math.max(0, alph2 + alph1 - C) // 14a
    H = Math.min(C, alph2 + alph1) // 14b
  }

  if (L === H) return false

  let k11 = await svm.polyKernel(x1, x1) // conveniences
  let k12 = await svm.polyKernel(x1, x2)
  let k22 = await svm.polyKernel(x2, x2)
  let eta = k11 + k22 - 2 * k12 // 15

  var a1
  var a2

  if (eta > 0) {
    a2 = alph2 - y2 * (e2 - e1) / eta // 16
    if (a2 >= H) {
      a2 = H // 17a
    } else if (a2 <= L) {
      a2 = L // 17b
    }
  } else { // "Under unusual circumstances, eta will not be positive"
    let f1 = y1 * (e1 + svm.bias) - alph1 * k11 - s * alph2 * k12 // 19a
    let f2 = y2 * (e2 + svm.bias) - alph2 * k22 - s * alph1 * k12 // 19b
    let l1 = alph1 + s * (alph2 - L) // 19c
    let h1 = alph1 + s * (alph2 - H) // 19d
    let lobj = (l1 * f1) + (L * f2) + (0.5 * l1 * l1 * k12) + (0.5 * L * L * k22) + (s * L * l1 * k12) // 19e
    let hobj = (h1 * f1) + (H * f2) + (0.5 * h1 * h1 * k11) + (0.5 * H * H * k22) + (s * H * h1 * k12) // 19f

    if (lobj < hobj - eps) {
      a2 = L
    } else if (lobj > hobj + eps) {
      a2 = H
    } else {
      a2 = alph2
    }
  }

  if (Math.abs(a2 - alph2) < eps * (a2 + alph2 + eps)) return false

  a1 = alph1 + s * (alph2 - a2) // 18

  // Update threshold (biasa). See section 2.3
  let b1 = e1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + svm.bias
  let b2 = e2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + svm.bias
  let newb

  if (a1 > 0 && C > a1) {
    newb = b1
  } else if (a2 > 0 && C > a2) {
    newb = b2
  } else {
    newb = (b1 + b2) / 2
  }

  let deltab = newb - svm.bias
  svm.bias = newb

  // Update error cache using new Lagrange multipliers
  let delta1 = y1 * (a1 - alph1)
  let delta2 = y2 * (a2 - alph2)

  for (let i = 0; i < xMatrix.length; i++) {
    if (svm.alpha[i] > 0 && svm.alpha[i] < C) {
      svm.errors[i] += delta1 * await svm.polyKernel(x1, xMatrix[i]) + delta2 * await svm.polyKernel(x2, xMatrix[i]) - deltab
    }
  }

  svm.errors[i1] = 0.0
  svm.errors[i2] = 0.0
  svm.alpha[i1] = a1 // Store a1 in the alpha array
  svm.alpha[i2] = a2 // Store a2 in the alpha array

  return true
}

svm.train = async function (xMatrix, yVector, maxIter) {
  let N = xMatrix.length
  svm.alpha = Array(xMatrix.length).fill(null)
  svm.errors = Array(xMatrix.length).fill(null)

  let numChanged = 0
  let examineAll = true
  let iter = 0

  while ((iter < maxIter && numChanged > 0) || (examineAll === true)) {
    iter++
    numChanged = 0
    if (examineAll === true) {
      // all training examples
      for (let i = 0; i < N; i++) {
        numChanged += await svm.examineExample(i, xMatrix, yVector)
      }
    } else {
      // examples where alpha is not 0 and not C
      for (let i = 0; i < N; i++) {
        if (svm.alpha[i] !== 0 && svm.alpha[i] !== svm.complexity) {
          numChanged += await svm.examineExample(i, xMatrix, yVector)
        }
      }
    }

    if (examineAll === true) {
      examineAll = false
    } else if (numChanged === 0) {
      examineAll = true
    }
  }

  let indicies = [] // indices of support vectors
  for (let i = 0; i < N; i++) {
    // Only store vectors with Lagrange multipliers > 0
    if (svm.alpha[i] > 0) indicies.push(i)
  }

  let numSuppVectors = indicies.length
  svm.weights = []
  for (let i = 0; i < numSuppVectors; i++) {
    let j = indicies[i]
    svm.supportVectors.push(xMatrix[j])
    svm.weights[i] = svm.alpha[j] * yVector[j]
  }
  svm.bias = -1 * svm.bias
  return iter
}

svm.examineExample = async function (i2, xMatrix, yVector) {
  // "Sequential Minimal Optimization: A Fast Algorithm for
  // Training Support Vector Machines", Platt, 1998.
  let C = svm.complexity
  let tol = svm.tolerance

  let x2 = xMatrix[i2] // "point" at i2
  let y2 = yVector[i2] // class label for p2
  let alph2 = svm.alpha[i2] || null // Lagrange multiplier for i2

  // SVM output on point[i2] - y2. (check in error cache)
  let e2
  if (alph2 > 0 && alph2 < C) {
    e2 = svm.errors[i2]
  } else {
    e2 = await svm.computeAll(x2, xMatrix, yVector) - y2
  }

  let r2 = y2 * e2
  if ((r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0)) {
    // See section 2.2
    var i1 = -1
    var maxErr = 0

    for (let i = 0; i < xMatrix.length; i++) {
      if (svm.alpha[i] > 0 && svm.alpha[i] < C) {
        let e1 = svm.errors[i]
        let delErr = Math.abs(e2 - e1)

        if (delErr > maxErr) {
          maxErr = delErr
          i1 = i
        }
      }
    }

    if (i1 >= 0 && await svm.takeStep(i1, i2, xMatrix, yVector)) return 1
    var rndi = svm.rnd(0, xMatrix.length)
    for (i1 = rndi; i1 < xMatrix.length; i1++) {
      if (svm.alpha[i1] > 0 && svm.alpha[i1] < C) {
        if (await svm.takeStep(i1, i2, xMatrix, yVector)) return 1
      }
    }

    for (i1 = 0; i1 < rndi; i1++) {
      if (await svm.alpha[i1] > 0 && svm.alpha[i1] < C) {
        if (await svm.takeStep(i1, i2, xMatrix, yVector)) return 1
      }
    }

    // "Both the iteration through the non-bound examples and the
    // iteration through the entire training set are started at
    // random locations"
    rndi = svm.rnd(0, xMatrix.length)
    for (i1 = rndi; i1 < xMatrix.length; i1++) {
      if (await svm.takeStep(i1, i2, xMatrix, yVector)) return 1
    }

    for (i1 = 0; i1 < rndi; i1++) {
      if (await svm.takeStep(i1, i2, xMatrix, yVector)) return 1
    }
  }

  // "In extremely degenerate circumstances, none of the examples
  // will make an adequate second example. When this happens, the
  // first example is skipped and SMO continues with another chosen
  // first example."
  return 0
}

async function test () {
  let leftPad = '          '

  console.log('\n**************************************************************************')
  console.log('Begin SVM from scratch JavaScript demo')
  console.log('Coded from MSDN article https://msdn.microsoft.com/en-us/magazine/mt833291')
  console.log('Support Vector Machines Using C# By James McCaffrey | March 2019')
  console.log('**************************************************************************\n\n')

  let trainX = []
  trainX[0] = [4, 5, 7]
  trainX[1] = [7, 4, 2]
  trainX[2] = [0, 6, 12]
  trainX[3] = [1, 4, 8]
  trainX[4] = [9, 7, 5]
  trainX[5] = [14, 7, 0]
  trainX[6] = [6, 9, 12]
  trainX[7] = [8, 9, 10]

  let trainY = []
  trainY[0] = -1
  trainY[1] = -1
  trainY[2] = -1
  trainY[3] = -1
  trainY[4] = 1
  trainY[5] = 1
  trainY[6] = 1
  trainY[7] = 1

  var lineOut = ''

  console.log('Training data:')
  for (let i = 0; i < trainX.length; i++) {
    lineOut = '[' + i + '] '
    for (let j = 0; j < trainX[i].length; j++) {
      lineOut += String(leftPad + trainX[i][j].toString()).slice(-6)
    }
    lineOut += '  |  ' + String(leftPad + trainY[i].toString()).slice(-3)
    console.log(lineOut)
  }

  console.log('\nCreating SVM with poly kernel degree = 2')
  console.log('Starting training...')
  let iter = await svm.train(trainX, trainY, svm.maxIter)
  console.log('Training complete in ' + iter + ' iterations\n')

  console.log('Support vectors:')
  lineOut = ''
  for (let i = 0; i < svm.supportVectors.length; i++) {
    lineOut += svm.supportVectors[i].toString() + '\n'
  }
  console.log(lineOut)

  console.log('Weights:')
  let lineOutW = ''
  for (let i = 0; i < svm.weights.length; i++) {
    lineOutW += svm.weights[i].toString() + ' '
  }
  console.log(lineOutW + '\n')

  console.log('\nBias = ' + svm.bias.toString() + '\n')

  for (let i = 0; i < trainX.length; i++) {
    let pred = await svm.computeDecision(trainX[i])
    console.log('Predicted decision value for [' + i + '] = ' + String(leftPad + pred).slice(-25))
  }

  let acc = await svm.accuracy(trainX, trainY)
  console.log('\nModel accuracy on test data = ' + acc.toString())

  let unknown = [3, 5, 7]
  let predDecVal = await svm.computeDecision(unknown)
  console.log('\nPredicted value for (3.0 5.0 7.0) = ' + predDecVal.toString())

  let predLabel = Math.sign(predDecVal)
  console.log('\nPredicted label for (3.0 5.0 7.0) = ' + predLabel)

  console.log('\nEnd demo ')
}

test()
