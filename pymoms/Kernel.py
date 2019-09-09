#  ==================================================================================
#  
#  Copyright (c) 2019, Evangelos G. Karakasis 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  
#  ==================================================================================
 
# TODO: 
# > Implement exact Legendre polynomials
# > Add references for the used polynomials
# > Apply (multiply) a gaussian function in the polynomial in order to 
#   give a localized nature (similar to Gabor wavelet).

import numpy as _np
# from abc import ABC as _ABC
# from abc import abstractmethod as _abstractmethod

# ===== POLYNOMIAL INTERFACE ========================================================

# # it would be useful to have some interface for enforcing specific
# # polynomial functionality and design. Currently, the following
# # code runs, but don't work (does not enforce the interface design).
#
# class _iPolynomial(_ABC):
#     @staticmethod
#     @_abstractmethod
#     def WeightScheme(upToDegree, *argv):
#         """Calculates polynomials weights. 

#             Parameters 
#             ---------- 
#             upToDegree: int
#                 The maximum degree for which we desire to calculate the polynomial values.

#             *argv: additional arguments
#                 Represents whatever parameter may be of need for specific weighting schemes. 

#             Returns 
#             ------- 
#             ndarray
#                 The resulted array is an 1d ndarray, where its rows represent a weight for 
#                 the corresponding polynomial values.
#         """
#         pass
    
#     @staticmethod
#     @_abstractmethod
#     def Poly(upToDegree, x):
#         """Calculates the n-th degree polynomial. 

#             Parameters 
#             ---------- 
#             upToDegree: int 
#                 The maximum degree for which we desire to calculate the polynomial values.

#             x: ndarray
#                 The variable of the polynomial. It must be a 1D ndarray of the disired length.

#             Returns
#             ------- 
#             ndarray
#                 The resulted array is a 2d ndarray, where the rows represent the degree
#                 and the columns the polynomial values. 

#             """
#         pass

# -----------------------------------------------------------------------------------

def GetKernel(family, dim):
    """It is responsible for returning the selected kernel's calculation function along with the coresponding weights and some other stuff.
    
    Parameters
    ----------
    family: str
        The 'family' parameter determines the kind of desired kernel that will be used in a moment transform. 

        Currently, the supported kernel families are:
            * 'chebyshev1': the Chebyshev polynomials of the first kind,        
            * 'chebyshevD': the Discrete Chebyshev (or else Tchebichef) polynomials,
            * 'geometric' : the geometric monomials (Pn(x)   = x**n),
            * 'central'   : the central monomials (Pn(x; xm) = (x-xm)**n)

    dim: int
        The 'dim' parameter represents the length of the set of values, which will be
        used as variable in the kernel's calculation.
        Example: in the geometric kernel (Pn(x)=x**n), the x variable takes values x = np.arange(0, dim)
    
    Returns
    -------
    tuple: (kernel, weights, xi, isOrthogonal)
        The first element of the resulted tuple is the kernel's function, the second the the coresponding
        weight function, the third is the variables values that are used as input in the kernel function
        and finally the fourth element is a flag (boolean) which infroms whether or not the kernel is orthogonal.
    """

    kernels = {
        'chebyshev1': { # continuous Chebychev of 1st kind
            'poly'        : ConOrthoPoly.Chebyshev1st.Poly,
            'weights'     : ConOrthoPoly.Chebyshev1st.WeightScheme,
            'x'           : _np.cos(_np.pi*(_np.arange(0, dim)+0.5)/float(dim)),
            'isOrthogonal': True,
        },
        'chebyshevD': { # discrete Chebyshef (or else Tchebichef)
            'poly'        : DisOrthoPoly.Chebyshev.Poly,
            'weights'     : DisOrthoPoly.Chebyshev.WeightScheme,
            'x'           : _np.arange(0, dim),
            'isOrthogonal': True,
        },
        'geometric': { 
            'poly'        : NonOrthoFunc.Geometric.Calc,
            'weights'     : NonOrthoFunc.Geometric.WeightScheme,
            'x'           : _np.arange(0, dim) / dim, # normalize for improve instability issues
            'isOrthogonal': False,
        },
        'central': { 
            'poly'        : NonOrthoFunc.Central.Calc,
            'weights'     : NonOrthoFunc.Central.WeightScheme,
            'x'           : (_np.arange(0, dim) - (dim-1)/2.) / dim, # normalize for improve instability issues
            'isOrthogonal': False,
        },
    }  

    kernel       = kernels[family]['poly']
    weights      = kernels[family]['weights']
    xi           = kernels[family]['x']
    isOrthogonal = kernels[family]['isOrthogonal']

    return kernel, weights, xi, isOrthogonal



# ===== DISCRETE ORTHOGONAL POLYNOMIALS =============================================

class DisOrthoPoly: # Orthogonal Discrete Polynomials
    class Chebyshev: # also known as Tchebichef ! 
        @staticmethod
        def WeightScheme(upToDegree, *argv):
            """Calculates weights for the Discrete Chebyshev polynomials. 

            Parameters 
            ---------- 
            upToDegree: int
                The maximum degree for which we desire to calculate the polynomial values.

            *argv: additional arguments
                It exists for sake of uniformity of similar functions. Here we don't need any 
                additional argument in order to calculate the weights.

            Returns 
            ------- 
            ndarray
                The resulted array is an 1d ndarray of shape (upToDegree, 1), where its rows 
                represent the weight for the corresponding polynomial values.

            Examples 
            -------- 
            upToDegree = 5
            W = OrthoDisPoly.Chebyshev.WeightScheme(upToDegree)
            """
            
            return _np.ones((int(upToDegree+1), 1))        
        
        @staticmethod
        def Poly(upToDegree, x):
            """Calculates up to n-th degree Discrete Chebyshev polynomial. 

            The discrete Chebyshev polynomials are orthogonal in the interval [0, N-1].

            Parameters 
            ---------- 
            upToDegree: int 
                The maximum degree for which we desire to calculate the polynomial values.
                
            x: ndarray
                The variable of the polynomial. The elements of x must belong to [0, N-1],
                where N determines the discrete domain in which the discrete Chebychef (or Tchebichef) 
                polynomials are defined.

            Returns
            ------- 
            ndarray
                The resulted array is a 2d ndarray of shape (upToDegree+1, N).

            """

            N = _np.max(x)+1
            P = _np.zeros((int(upToDegree+1), int(N)))
            
            plnml = DisOrthoPoly.Chebyshev._Poly
            for degree in range(0, int(upToDegree+1)):
                P[degree, :] = plnml(degree, N)[0,:]
                
            return P
                    
        @staticmethod
        def _Poly(degree, N):
            """Calculates the n-th degree Discrete Chebyshev polynomial. 

            The discrete Chebyshev polynomials are orthogonal in the interval [0, N-1].

            Parameters 
            ---------- 
            degree: int 
                The degree for which we desire to calculate the polynomial values.
                
            N: int
                Determines the discrete domain in which the discrete Chebychef (or Tchebichef) 
                polynomials are defined. The variable X of the polynomial must belong in the 
                interval [0, N-1].

            Returns
            ------- 
            ndarray
                The resulted array is a 1d ndarray of shape (1, N).
            """
            
            X = N-1
            P = _np.zeros((1, N))
            
            T00 = 1.0 / _np.sqrt(N)
            Tn0 = T00
            for n in _np.arange(1, degree+1):
                Tn0 = - _np.sqrt( (N - n) / float(N + n) ) * _np.sqrt( (2.0 * n + 1.0) / (2.0 * n - 1.0) ) * Tn0                            
            
            if X == 0:
                P[0, 0] = Tn0
            elif X == 1: 
                P[0, 0] = Tn0
                P[0, 1] = (1.0 + degree * (1.0 + degree) / float(1.0 - N)) * Tn0
            else:
                P[0, 0] = Tn0
                P[0, 1] = (1.0 + degree * (1.0 + degree) / float(1.0 - N)) * Tn0
                
                for x0 in range(2, X - 1):
                    if x0 > N/2:
                        x = N - 1 - x0
                        
                        r1 = ( - degree * (degree + 1.0) - (2.0 * x - 1.0) * (x - N - 1.0) - x ) / float( x * (N - x) )
                        r2 = ( (x - 1.0) * (x - N - 1.0) ) / float( x * (N - x) )
            
                        P[0, x0] = ((-1)**degree) * (r1 * P[0, x - 1] + r2 * P[0, x - 2])
                    else:
                        x = x0
                        
                        r1 = ( - degree * (degree + 1.0) - (2.0 * x - 1.0) * (x - N - 1.0) - x ) / float( x * (N - x) )
                        r2 = ( (x - 1.0) * (x - N - 1.0) ) / float( x * (N - x) )
                        
                        P[0, x0] = r1 * P[0, x - 1] + r2 * P[0, x - 2]
                        
                P[0, X - 1] = ((-1)**degree) * P[0, 1];
                P[0, X] = ((-1)**degree) * P[0, 0]
                
            return P
            


# ===== CONTINUOUS ORTHOGONAL POLYNOMIALS ===========================================

class ConOrthoPoly: # Orthogonal Continuous Polynomials
    class Chebyshev1st:
        @staticmethod
        def WeightScheme(upToDegree, *argv):
            """Calculates weights for the Chebyshev polynomial of the first kind. 

            Parameters 
            ---------- 
            upToDegree: int
                The maximum degree for which we desire to calculate the polynomial values.

            *argv: additional arguments
                It exists for sake of uniformity of similar functions. Here we need an additional 
                parameter with which is defined in the following lines:

                xLen: int
                    The length of the polynomial's variable domain. 
                    e.g. if we want to calculate the Chebyshev1 polynomial in the range x=np.linspace(-1, 1, 11)
                    the xLen parameter should have the value 11.     

            Returns 
            ------- 
            ndarray
                The resulted array is an 1d ndarray, where its rows represent a weight for 
                the corresponding polynomial values.

            Examples 
            -------- 
            x = np.linspace(-1, 1, 11)
            upToDegree = 5
            W = ConOrthoPoly.Chebyshev1st.WeightScheme(upToDegree, len(x))
            """
            
            xLen = float(argv[0])
            W = _np.ones((upToDegree + 1, 1)) * 2. / xLen
            W[0,0] = 1. / xLen
            return W

        @staticmethod
        def Poly(upToDegree, x):
            """Calculates the n-th degree Chebyshev polynomial of first kind. 

            The continuous Chebyshev polynomials are orthogonal in the interval [−1, 1].

            Parameters 
            ---------- 
            upToDegree: int 
                The maximum degree for which we desire to calculate the polynomial values (P_{n}(x)).

            x: ndarray
                The variable of the polynomial. The elements of x must belong to [-1, 1].

            Returns
            ------- 
            ndarray
                The resulted array is a 2d ndarray, where the rows represent the degree
                and the columns the polynomial values. 

            Examples 
            -------- 
            x = np.linspace(-1, 1, 11)
            upToDegree = 5
            Pval = ConOrthoPoly.Chebyshev1st.Poly(upToDegree, x)
            """

            m = len(x)
            if upToDegree < 0:
                raise ValueError('The parameter upToDegree must be >= 0.')

            v = _np.zeros( (upToDegree + 1, m), float)

            v[0, :] = 1.0

            if upToDegree < 1:
                return v

            x = x.flatten()

            v[1, :] = x

            for i in _np.arange(1, upToDegree):
                v[i + 1, :] = 2.0 * x * v[i, :] - v[i - 1, :]

            return v



# ===== NON ORTHOGONAL FUNCTIONS ====================================================

class NonOrthoFunc:
    class Geometric:
        @staticmethod
        def WeightScheme(upToDegree, *argv):
            """Calculates weights for the monomial f(x; n) = x^n. 

            Parameters 
            ---------- 
            upToDegree: int
                The maximum degree for which we desire to calculate the polynomial values.

            *argv: additional arguments
                It exists for sake of uniformity of similar functions. Here we don't need any 
                additional argument in order to calculate the weights.

            Returns 
            ------- 
            ndarray
                The resulted array is an 1d ndarray of shape (upToDegree, 1), where its rows 
                represent the weight for the corresponding polynomial values.

            Examples 
            -------- 
            upToDegree = 5
            W = NonOrthoFunc.Geometric.WeightScheme(upToDegree)
            """
            
            return _np.ones((int(upToDegree+1), 1))   


        @staticmethod
        def Calc(upToDegree, x):
            """Calculates up to n-th degree geometric monomial f(x; n) = x^n. 

            The the particular function is defined in the interval [0, N-1].

            Parameters 
            ---------- 
            upToDegree: int 
                The maximum degree for which we desire to calculate the geometric monomial values.
                
            x: ndarray
                The variable of the geometric monomial. The elements of x must belong to [0, N-1],
                where N determines the discrete domain in which the geometric monomial is defined.

                Note that to improve instability issues we may normalize the x by dividing with N.

            Returns
            ------- 
            ndarray
                The resulted array is a 2d ndarray of shape (upToDegree+1, N).

            """

            N = len(x)
            P = _np.zeros((int(upToDegree+1), int(N)))
            
            for degree in range(0, int(upToDegree+1)):
                P[degree, :] = x**degree
                
            return P

    class Central:
        @staticmethod
        def WeightScheme(upToDegree, *argv):
            """Calculates weights for the monomial f(x; n) = (x-xm)^n. 

            Parameters 
            ---------- 
            upToDegree: int
                The maximum degree for which we desire to calculate the polynomial values.

            *argv: additional arguments
                It exists for sake of uniformity of similar functions. Here we don't need any 
                additional argument in order to calculate the weights.

            Returns 
            ------- 
            ndarray
                The resulted array is an 1d ndarray of shape (upToDegree, 1), where its rows 
                represent the weight for the corresponding polynomial values.

            Examples 
            -------- 
            upToDegree = 5
            W = NonOrthoFunc.Central.WeightScheme(upToDegree)
            """
            
            return _np.ones((int(upToDegree+1), 1))   


        @staticmethod
        def Calc(upToDegree, x_):
            """Calculates up to n-th degree central monomial f(x; n) = (x-xm)^n. 

            Parameters 
            ---------- 
            upToDegree: int 
                The maximum degree for which we desire to calculate the central monomial values.
                
            x_: ndarray
                For the parameter x_ holds that x_ = x-xm, where x is the variable of the central monomial
                that belongs to [0, N-1], the xm, 0<xm<N, is a value that shifts the center of x and N determines 
                the discrete domain in which the variable x takes values. Thus, the parameters x_ belongs to
                [-xm, N-1-xm].

                Note that to improve instability issues we may normalize the _x by deviding with N.


            Returns
            ------- 
            ndarray
                The resulted array is a 2d ndarray of shape (upToDegree+1, N).

            """

            N = len(x_)
            P = _np.zeros((int(upToDegree+1), int(N)))
            
            for degree in range(0, int(upToDegree+1)):
                P[degree, :] = x_**degree
                
            return P