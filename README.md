# Moment-Based Transforms (pymoms)

This project includes code that was primarily developed (though in c++) for my [PhD](http://thesis.ekt.gr/thesisBookReader/id/31558#page/1/mode/2up) in order to apply spectral expansions, which use as kernels special functions such as (non)orthogonal polynomials (e.g. Chebyshev of the first kind, discrete Chebyshev, Zernike, orthogonal Fourier-Mellin, etc), instead of exponential functions (like in Fourier transform).

Currently, the library supports the traditional 2D Cartesian geometric moments (they are also called raw moments), the central moments, the Chebyshev of the first kind and the discrete Chebyshev (or Tchebichef) ones. In the near future I aim to add functionality regarding:

* 1D moment functions,
* Cartesian 2D complex moment functions, 
* 2D radial moment functions and their invariant counterparts,
* Hu, normalized central and standard geometric moment functions,
* 1D & 2D signal (image) decomposition using moment functions.



## References

Here you can find some theory, which may help you to get deeper in your understanding on the field of moment-based spectral expansions.

[1] Mukundan, R., Ong, S. H., & Lee, P. A. (2001). Image analysis by Tchebichef moments. *IEEE Transactions on image Processing*, *10*(9), 1357-1364.

[2] Papakostas, G. A., Koulouriotis, D. E., & Karakasis, E. G. (2009). A unified methodology for the efficient computation of discrete orthogonal image moments. *Information Sciences*, *179*(20), 3619-3633.

[3] Xiao, B., Ma, J. F., & Cui, J. T. (2010, October). Invariant pattern recognition using radial Tchebichef moments. In *2010 Chinese Conference on Pattern Recognition (CCPR)* (pp. 1-5). IEEE.

[4] Karakasis, E. G., Papakostas, G. A., Koulouriotis, D. E., & Tourassis, V. D. (2013). A unified methodology for computing accurate quaternion color moments and moment invariants. *IEEE Transactions on Image Processing*, *23*(2), 596-611.



## Getting Started

### Dependencies

The only dependency, currently, is the numpy package. To install it just use something like

> $ pip install numpy

or 

> $ conda install numpy

### Installation

To install this package just download this repository from GitHub or by using the following command line:

> $ git clone https://github.com/ekarakasis/pymoms

Afterwards, go to the local root folder, open a command line and run:

> $ pip install .

and if you want to install it to a specific Anaconda environment then write:

> $ activate <Some_Environment_Name>
>
> $ pip install .

### How to Uninstall the package

To uninstall the package just open a command and write:

> $ pip unistall pymoms

To uninstall it from a specific conda environment write:

> $ activate <Some_Environment_Name>
>
> $ pip unistall pymoms



## Examples

### Example  01

```python
from pymoms.MomTransform import Moments2D

# creates a small 2D matrix with integer values in the range [0, 255]    
# for easy visualization of the results
lngth = 5
Mtx2D = np.random.randint(0, 255, (lngth, lngth))

# defines a class instance, which use as kernel on both axes the Chebyshev
# polynomial of the first kind. For achieving a perfect reconstruction
# we will need lngth x lngth coefficients, so the upToDegree parameter 
# must be equal to lngth-1, since the first degree is 0.
MT = Moments2D(
    family=['chebyshev1'], 
    upToDegree=[lngth-1], 
    shape=Mtx2D.shape
) # moment transform instance

coeffs = MT.ForwardTransform(Mtx2D)
Mtx2D_reconstructed = MT.InverseTransform(coeffs)

print('original 2D matrix:\n {}'.format(Mtx2D))
print('\nreconstructed 2D matrix:\n {}'.format(Mtx2D_reconstructed))

# NOTE: not all the kernels have inverse transform.            
```
Output:
```
    original 2D matrix:
     [[ 97 139 232 229   6]
     [141 196 151 214 213]
     [ 40  38  67 189  22]
     [ 82 132   5 186 157]
     [ 14 149 188  91  75]]
    
    reconstructed 2D matrix:
     [[ 97. 139. 232. 229.   6.]
     [141. 196. 151. 214. 213.]
     [ 40.  38.  67. 189.  22.]
     [ 82. 132.   5. 186. 157.]
     [ 14. 149. 188.  91.  75.]]
```



### Example 02

```python
from pymoms import Kernel
import matplotlib.pyplot as plt
import numpy as np

def myplot(x, y, title, ylabel):
    plt.figure(figsize=(15,5))
    plt.plot(x, y)
    plt.grid('on')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel(ylabel)
    plt.show()
    
upToOrder = 5
xLen = 100

# Get the Chebyshev Polynomial of the first kind as Kernel.
# Actually, for the particular polynomial family, the resulted xi is given by:
# xi = np.cos(np.pi*(np.arange(0, xLen)+0.5)/float(xLen))
kernel, weights, xi, isOrthogonal = Kernel.GetKernel('chebyshev1', xLen)

# Currently, the supported kernel families are:
#    * 'chebyshev1': the Chebyshev polynomials of the first kind,        
#    * 'chebyshevD': the Discrete Chebyshev (or else Tchebichef) polynomials,
#    * 'geometric' : the geometric monomials (Pn(x)   = x**n),
#    * 'central'   : the central monomials (Pn(x; xm) = (x-xm)**n)

# Calculate the polynomial values and the corresponding weights
P = kernel(upToOrder, xi)
Wp = weights(upToOrder, xLen)

P_norm = np.zeros(P.shape)
# normalized the polynomial values using the weights
for idx, p in enumerate(P):
    P_norm[idx,:] = P[idx,:] * Wp[idx]

myplot(xi, P.T, 'Chebyshef Polynomial of first kind', '$P_{n}(x)$')
myplot(xi, P_norm.T, 'Norm. Chebyshef Polynomial of first kind', '$\overline{P}_{n}(x)$')

# let us define a different xi:
xi = np.linspace(-1, 1, xLen)
myplot(xi, P.T, 'Chebyshef Polynomial of first kind', '$P_{n}(x)$')
```
Output:
![png](images/output(1).png)

![png](images/output(2).png)

![png](images/output(3).png)

## License

This project is licensed under the MIT License.

