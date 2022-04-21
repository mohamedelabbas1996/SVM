from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers
def SvmOptim(X, y):
  m,n = X.shape
  y = y.reshape(-1,1) * 1.
  X_dash = y * X
  H = np.dot(X_dash , X_dash.T) * 1

  #Converting into cvxopt format
  P = cvxopt_matrix(H)
  q = cvxopt_matrix(-np.ones((m, 1)))
  G = cvxopt_matrix(-np.eye(m))
  h = cvxopt_matrix(np.zeros(m))
  A = cvxopt_matrix(y.reshape(1, -1))
  b = cvxopt_matrix(np.zeros(1))
  solution = solvers.qp(P, q, G, h, A, b)
  alpha = np.array(solution['x'])
  w = ((alpha * y).T @X).reshape(-1, 1)
  seuil = 1e-6
  s = (alpha > seuil).flatten()
  b = y[s] - np.dot(X[s], w)
  print(alpha)
  print('w = ', w.flatten())
  print('b = ', b[0])
  return w, b 
