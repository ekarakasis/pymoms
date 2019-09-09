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
# > Add logger
# > Implement Hu, normalized central and standard geometric moments
# > Implement complex moments and their invariant counterparts
# > Implement radial moments
# > Implement radial moment invariants
# > Implement radial quaternionic color moments and moment invariants
# > Apply moments transform for image decomposition

# reference from Greek National Archive of PhD Theses:
# http://thesis.ekt.gr/thesisBookReader/id/31558#page/26/mode/2up

import sys 
sys.path.append('../')
# sys.path.append('../../')

import numpy as _np
from pymoms import Kernel as _Kernel

class Moments2D: 
    """Calculates 2D Cartisian moment functions.
    
    Parameters 
    ----------
    family: list of str, ['<kernel of Y axis>', '<kernel of X axis>']
        Normally the 'family' parameter must be a list consisted of two elements.
        These elements determine the kind of kernels on each axis. The first element
        determines the family of the kernel used in the Y axis (that is the rows of the 2D matrix)
        and the second element determines the family of the kernel of the X axis (that is
        the columns of the 2D matrix). Most of the time the two elements should indicate
        the same kernel family (so it is also allowed to feed a list with only one element indicating
        the same kernel family for both axes), but due to the separable nature of the
        particular transform there is also possible to use different kernels, as well.

        Currently, the supported kernel families are:
            * 'chebyshev1': the Chebyshev polynomials of the first kind,        
            * 'chebyshevD': the Discrete Chebyshev (or else Tchebichef) polynomials,
            * 'geometric' : the geometric monomials (Pn(x)   = x**n),
            * 'central'   : the central monomials (Pn(x; xm) = (x-xm)**n)

    upToDegree: list of int
        A list with two integer values that indicate the maximum degree (or order) for 
        each kernel. The first value corresponds to the maximum degree of the kernel 
        used in the Y axis, while the second element corersponds to the degree of the kernel
        used in the X axis. Similarly with the case of 'family' parameter, when the 
        used values are the same, it is allowed to feed a list with only one integer value.

    shape: tuple of int
        The 'shape' parameter represents the actual shape of the 2D matrix for which
        we want to calculate the particular transform's coefficients.


    Examples
    --------
    from pymoms.MomTransform import Moments2D

    # creates a small 2D matrix with integer values in the range [0, 255]    
    # for easy visualization of the results
    dim = 5
    Mtx2D = np.random.randint(0, 255, (dim, dim))

    # defines a class instance, which use as kernel on both axes the Chebyshev
    # polynomial of the first kind. For achieving a perfect reconstruction
    # we will need dim x dim coefficients, so the upToDegree parameter 
    # must be equal to dim-1, since the first degree is 0.
    MT = Moments2D(['chebyshev1'], [dim-1], Mtx2D.shape) # moment transform instance

    coeffs = MT.ForwardTransform(Mtx2D)
    Mtx2D_reconstructed = MT.InverseTransform(coeffs)

    print(Mtx2D)
    print(Mtx2D_reconstructed)

    # NOTE: not all the kernels have inverse transform.


    References     
    ----------
    Here you can find some theory, which may help you to get deeper in your understanding 
    on the field of moment-based spectral expansions.

    [1] Mukundan, R., Ong, S. H., & Lee, P. A. (2001). Image analysis by Tchebichef 
        moments. IEEE Transactions on image Processing, 10(9), 1357-1364.

    [2] Papakostas, G. A., Koulouriotis, D. E., & Karakasis, E. G. (2009). A unified 
        methodology for the efficient computation of discrete orthogonal image moments. 
        Information Sciences, 179(20), 3619-3633.

    [3] Xiao, B., Ma, J. F., & Cui, J. T. (2010, October). Invariant pattern recognition 
        using radial Tchebichef moments. In 2010 Chinese Conference on Pattern Recognition 
        (CCPR) (pp. 1-5). IEEE.

    [4] Karakasis, E. G., Papakostas, G. A., Koulouriotis, D. E., & Tourassis, V. D. (2013). 
        A unified methodology for computing accurate quaternion color moments and moment 
        invariants. IEEE Transactions on Image Processing, 23(2), 596-611.
    """

    def __init__(self, family, upToDegree, shape):

        # ===== Check parameters length =====
        if len(family) == 1:
            family.append(family[0])

        if len(upToDegree) == 1:
            upToDegree.append(upToDegree[0])


        # ===== Keep the inputs =====
        self._family     = family
        self._upToDegree = upToDegree
        self._shape      = shape


        # ===== Initialize Kernels & Weights =====                        
        rows = 0
        cols = 1

        upToDegreeY = upToDegree[rows]
        upToDegreeX = upToDegree[cols]
        DimY        = shape[rows]
        DimX        = shape[cols]

        kernelY, weightsY, yj, self._isOrthogonalY = _Kernel.GetKernel(family[rows], DimY)
        kernelX, weightsX, xi, self._isOrthogonalX = _Kernel.GetKernel(family[cols], DimX)                
                        
        self._krnY = kernelY(upToDegreeY, yj)
        self._krnX = kernelX(upToDegreeX, xi)
        self._wy   = weightsY(upToDegreeY, DimY)
        self._wx   = weightsX(upToDegreeX, DimX)

        # # used by the implementation of forward transfrom with convolution
        # self._krnl2D = _np.zeros((self._upToDegree[0]+1, self._upToDegree[1]+1, self._shape[0], self._shape[1]))
        # for m in range(0, self._upToDegree[0]+1):
        #     for n in range(0, self._upToDegree[1]+1):
        #         self._krnl2D[m, n, :, :] = (self._wy[m, 0] * self._wx[n, 0]) * (self._krnY[m, :, _np.newaxis].dot(self._krnX[n, :, _np.newaxis].T))
        
        
    def ForwardTransform(self, Mtx2D):  
        """The forward moment transform.
        
        Parameters
        ----------
        Mtx2D : ndarray
            As indicated by its name, the input must be a 2D numpy array (not numpy matrix!).
        
        Returns
        -------
        ndarray
            The function returns a 2D ndarray, the shape of which is determined by the inputs
            that have been given in the class instance definition.
        """  

        # NOTE: for template matching it is a good idea to enable the following code,
        # but in cases where the moments are going to be used as local descriptors
        # of key points, should be left as comment.

        # it calculates the center of mass for being used in central moments instead
        # of the middle point of the kernel's domain.
        # if self._family[0] == 'central' and self._family[1] == 'central':
        #     ym, xm = self.GetCenterOfMass(Mtx2D)
        #     yj = _np.arange(0, self._shape[0]) - ym
        #     xi = _np.arange(0, self._shape[1]) - xm
        #     Calc = _Kernel.NonOrthoFunc.Central.Calc
        #     self._krnY = Calc(self._upToDegree[0], yj)
        #     self._krnX = Calc(self._upToDegree[1], xi)


        return (self._wy.dot(self._wx.T)) * (self._krnY.dot((self._krnX.dot(Mtx2D.T)).T))


    # -------------------------------------------------------------------------------    
    # # NOT very FAST implementation !!!
    # # it is written only to present an alternative implementation 
    # # that me be useful to know about.

    # from scipy import signal as _spSignal
    # def ForwardTransformC(self, Mtx2D):  
    #     """The forward moment transform using 2d correlation.
        
    #     Parameters
    #     ----------
    #     Mtx2D : ndarray
    #         As indicated by its name, the input must be a 2D numpy array (not numpy matrix!).
        
    #     Returns
    #     -------
    #     ndarray
    #         The function returns a 2D ndarray, the shape of which is determined by the inputs
    #         that have been given in the class instance definition.
    #     """  

        
    #     Mnm = _np.zeros((self._upToDegree[0]+1, self._upToDegree[1]+1))
    #     for m in range(0, self._upToDegree[0]+1):
    #         for n in range(0, self._upToDegree[1]+1):
    #             krnl2D = (self._wy[m, 0] * self._wx[n, 0]) * (self._krnY[m, :, _np.newaxis].dot(self._krnX[n, :, _np.newaxis].T))
    #             Mnm[m, n] =  _spSignal.correlate2d(Mtx2D, krnl2D, mode='valid')[0]
    #             # Mnm[m, n] =  _spSignal.correlate2d(Mtx2D, self._krnl2D[m, n, :, :], mode='valid')[0]
    #     return Mnm
        # -------------------------------------------------------------------------------
    
    
    def InverseTransform(self, coeffs2D):     
        """The inverse moment transform.
        
        Parameters
        ----------
        coeffs2D : ndarray
            The input of the function is a 2D ndarray with the coefficients that have
            been calculated by the forward moment transform.
            
        
        Returns
        -------
        ndarray
            The function returns a 2D ndarray with the same shape to the one that have been 
            used as input in the forward transform. In the case where the number of coefficients
            calculated with the forward transform is the same with the number of te Mtx2D elements,
            the reconstruction process (inverse transform) is perfect. 

            Note: not all the kernel families can lead to (perfect) reconstruction. Usually, the
            best kernels must be orthogonal.
        """
        
        if self._isOrthogonalY and self._isOrthogonalX:
            return (coeffs2D.T.dot(self._krnY)).T.dot(self._krnX) 
        else:
            return None 


    @staticmethod
    def GetCenterOfMass(Mtx2D):
        """Calculates the center of mass of a 2D ndarray.
        
        Parameters
        ----------
        Mtx2D: ndarray
            The 2D array for which we want to find the center of mass.
        
        Returns
        -------
        tuple
            The coordinates of the center of mass.
        """
        y, x = _np.mgrid[:Mtx2D.shape[0], :Mtx2D.shape[1]]
        ym   = _np.sum(y * Mtx2D) / _np.sum(Mtx2D)
        xm   = _np.sum(x * Mtx2D) / _np.sum(Mtx2D)
        return ym, xm


    @staticmethod
    def GetOrientation(Mtx2D):
        """Calculates the orientation of the input 2D array.
        
        Parameters
        ----------
        Mtx2D: ndarray
            The 2D array for which we want to find its orientation.
        
        Returns
        -------
        float
            The angle in the range [0, 2*np.pi)
        """

        y, x = _np.mgrid[:Mtx2D.shape[0], :Mtx2D.shape[1]]
        ym   = _np.sum(y * Mtx2D) / _np.sum(Mtx2D)
        xm   = _np.sum(x * Mtx2D) / _np.sum(Mtx2D)
        
        mu11 = _np.sum((y - ym) * (x - xm) * Mtx2D)
        mu02 = _np.sum(((y - ym)**2) * Mtx2D) 
        mu20 = _np.sum(((x - xm)**2) * Mtx2D) 
        mu30 = _np.sum(((x - xm)**3 ) * Mtx2D) 

        # the angle theta is defined in the range [-pi/4, pi/4]
        mud = mu20 - mu02
        theta = 0.5 * _np.arctan(2 * mu11 / mud)

        # for calculating the theta in the range [-pi, pi]
        # the following code should take place.        
        if (mu11 == 0) and (mud < 0) and (theta == 0):
            theta = _np.pi / 2.
        elif (mu11 > 0) and (mud < 0) and (theta > -_np.pi/4.) and (theta < 0):
            theta = theta + _np.pi / 2.
        elif (mu11 > 0) and (mud == 0) and (theta == 0):
            theta = _np.pi / 4.
        elif (mu11 < 0) and (mud == 0) and (theta == 0):
            theta = -_np.pi / 4.
        elif (mu11 < 0) and (mud < 0) and (theta > 0) and (theta < _np.pi/4.):
            theta = theta - _np.pi / 2.

        if mu30 < 0:
            if theta < 0:
                theta = theta + _np.pi 
            else: 
                theta = theta - _np.pi 


        return theta