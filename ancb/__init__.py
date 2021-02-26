from numpy import ndarray, asarray, empty, empty_like  # type: ignore
from typing import Tuple, NoReturn
from typing import Union

from functools import reduce
from itertools import zip_longest, starmap

import operator

from numpy import (
    matmul, add, subtract, multiply, divide, mod, floor_divide, power,
    negative, positive, absolute,
    right_shift, left_shift, bitwise_and, invert, bitwise_or, bitwise_xor
)

import numpy as np


def can_broadcast(shape1, shape2) -> bool:
    """
    Check if shapes shape1 and shape2 can be broadcast together.
    shape1 and shape2 are tuples representing the shapes of two ndarrays.

    :param Tuple shape1: first shape to parse
    :param Tuple shape2: second shape to parse

    :rtype: bool
    """
    return(
        reduce(
            lambda a, b: a and b,
            starmap(
                lambda a, b: (a == b or (a == 1 or b == 1)),
                zip_longest(shape1, shape2, fillvalue=1)
            )
        )
    )


def star_can_broadcast(starexpr) -> bool:
    """
    Check if shapes shape1 and shape2 can be broadcast together from a
    tuple of zip_longest(shape1, shape2, fillvalue=1) called the "starexpr"

    :param Tuple starexpr: starexpr to parse

    :rtype: bool
    """

    return (
        reduce(
            lambda a, b: a and b,
            starmap(
                lambda a, b: (a == b or (a == 1 or b == 1)),
                starexpr
            )
        )
    )


class NumpyCircularBuffer(ndarray):
    """
    Implements a circular (ring) buffer using a numpy array. This
    implmentation uses an internal size count and capacity count so that
    the data region is fully utilized.
    """

    def __new__(cls, data, bounds: Tuple[int, int] = (0, 0)):
        """
        Generate a circular buffer over existing array. The dimension 0 is
        used to index elements of the buffer.

        For example, for an ndim array of shape (N, a, b, c), the size of the
        buffer is interpretted to be N and the elements are arrays of shape
        (a, b, c).

        :param data: Data backing the buffer. Interpretted as a numpy array.

        :param Tuple[int, int] bounds: tuple of (begin, end) indices expressing
        where the buffer begins and ends over the data. For example, for a
        buffer [0, 1, 2, -1, -1] choosing (0, 2) would say [0, 1, 2] is in the
        buffer. For [1, 2, -1, -1, 0] choosing (4, 2) would select [0, 1, 2].
        """
        obj = asarray(data).view(cls)

        obj._begin = bounds[0]
        obj._end = bounds[1]
        obj._capacity = obj.shape[0]

        if (obj._begin < obj._end):
            obj._size = (obj._end - obj._begin) + 1
        elif (obj._begin > obj._end):
            # ((obj._capacity - 1) - obj._begin) + obj._end + 1
            obj._size = obj._capacity - obj._begin + obj._end
        else:
            obj._size = 0

        return (obj)

    def __init__(self, *args, **kwargs):
        # This is here for typing info for linters, not anything else
        super().__init__()
        self._begin: int
        self._end: int
        self._capacity: int

    def __matmul__(self, x):
        x = asarray(x)

        if (x.ndim == 0):
            ValueError(
                "matmul: Input operand 1 does not have enough dimensions "
                "(has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) "
                "requires 1"
            )

        if x.ndim == 1 and self.ndim == 1:
            # Dot product
            if x.shape[0] == self._size:
                if self.fragmented:
                    k = self._capacity - self._begin  # fragmentation index

                    out = matmul(self[self._begin:], x[:k]).view(ndarray)
                    out += matmul(self[:self._end], x[k:]).view(ndarray)
                else:
                    if self._begin < self._end:
                        part = self[self._begin:self._end]
                    elif self._end == 0:
                        part = self[self._begin:]

                    out = matmul(part, x).view(ndarray)

                return(out)
            else:
                raise ValueError(
                    "matmul: Input operand 1 has a mismatch in its core "
                    "dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)"
                    " (size {n} is different from {m})".format(
                        n=self._size,
                        m=x.shape[0]
                    )
                )
        elif self.ndim == 1 and x.ndim > 1:
            if self._size == x.shape[-2]:
                out = empty(*x.shape[:-2], x.shape[-1])

                if self.fragmented:
                    k = self._capacity - self._begin  # fragmentation index

                    matmul(self[self._begin:], x[..., :k, :], out)
                    out += matmul(self[:self._end], x[..., k:, :]).view(
                        ndarray
                    )
                else:
                    if self._begin < self._end:
                        part = self[self._begin:self._end]
                    elif self._end == 0:
                        part = self[self._begin:]

                    matmul(part, x, out).view(ndarray)

                return(out)
            else:
                raise ValueError(
                    "matmul: Input operand 1 has a mismatch in its core "
                    "dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)"
                    " (size {n} is different from {m})".format(
                        n=self.shape[-2],
                        m=x.shape[0]
                    )
                )
        elif self.ndim == 2:
            if (self.shape[-1] == x.shape[-2]):
                out = empty(
                    (*x.shape[:-2], self.shape[-1], x.shape[-2])
                )

                if self.fragmented:
                    k = self._capacity - self._begin  # fragmentation index

                    matmul(self[self._begin:], x, out[..., :k, :])
                    matmul(self[:self._end], x, out[..., k:, :])
                else:
                    if self._begin < self._end:
                        part = self[self._begin:self._end]
                    elif self._end == 0:
                        part = self[self._begin:]

                    matmul(part, x, out)

                return(out.view(ndarray))

            else:
                raise ValueError(
                    (
                        "matmul: Input operand 1 has a mismatch in its core "
                        "dimension 0, with gufunc signature (n?,k),(k,m?)->"
                        "(n?,m?) (size {n} is different from {m})"
                    ).format(
                        n=self.shape[-1],
                        m=x.shape[-2]
                    )
                )
        else:
            if (self.shape[-1] == x.shape[-2]):
                self_shape = (self._size, *self.shape[1:-2])

                starexpr = tuple(
                    zip_longest(self_shape, x.shape[:-2], fillvalue=1)
                )
                if star_can_broadcast(starexpr):
                    broadcast_shape = tuple(
                        starmap(
                            lambda a, b: max(a, b),
                            starexpr
                        )
                    )

                    out = empty(
                        (*broadcast_shape, self.shape[-2], x.shape[-1])
                    )

                    if self.fragmented:
                        k = self._capacity - self._begin  # fragmentation index

                        if x.ndim > 2:
                            matmul(self[self._begin:], x[:k], out[:k])
                            matmul(self[:self._end], x[k:], out[k:])
                        else:
                            matmul(self[self._begin:], x, out[:k])
                            matmul(self[:self._end], x, out[k:])
                    else:
                        if self._begin < self._end:
                            part = self[self._begin:self._end]
                        elif self._end == 0:
                            part = self[self._begin:]

                        matmul(part, x, out)

                    return(out.view(ndarray))
                else:
                    raise ValueError(
                        (
                            "operands could not be broadcast together with"
                            "remapped shapes [original->remapped]: "
                            "{shape_b}->({shape_bn}, newaxis,newaxis) "
                            "{shape_a}->({shape_an}, newaxis,newaxis) "
                            "and requested shape ({n},{m})"
                        ).format(
                            shape_a=self_shape,
                            shape_b=x.shape,
                            shape_an=self.shape[:-2].__str__()[:-1],
                            shape_bn=x.shape[:-2].__str__()[:-1],
                            n=self.shape[-1],
                            m=x.shape[-2]
                        )
                    )
            else:
                raise ValueError(
                    (
                        "matmul: Input operand 1 has a mismatch in its core "
                        "dimension 0, with gufunc signature (n?,k),(k,m?)->"
                        "(n?,m?) (size {n} is different from {m})"
                    ).format(
                        n=self.shape[-1],
                        m=x.shape[-2]
                    )
                )

    def matmul(self, x, work_buffer):
        """
        Performs buffer @ x. For matmul, extra space will be needed to
        perform the operation if

        buffer.fragmented == True and buffer.ndim == 1

        :param array_like x: Array to be multiplied by.
        :param ndarray work_buffer:
            Extra preallocated space to be used. It must be the same datatype
            as the output. While the shape of the work_buffer not matter, it
            must have space for at least as many elements as the output.
        :rtype: ndarray
        """

        x = asarray(x)
        space = work_buffer.flat

        if (x.ndim == 0):
            ValueError(
                "matmul: Input operand 1 does not have enough dimensions "
                "(has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) "
                "requires 1"
            )

        if x.ndim == 1 and self.ndim == 1:
            # Dot product
            if x.shape[0] == self._size:
                if self.fragmented:
                    k = self._capacity - self._begin  # fragmentation index

                    out = matmul(self[self._begin:], x[:k]).view(ndarray)
                    out += matmul(self[:self._end], x[k:])
                else:
                    if self._begin < self._end:
                        part = self[self._begin:self._end]
                    elif self._end == 0:
                        part = self[self._begin:]

                    out = matmul(part, x).view(ndarray)

                return(out)
            else:
                raise ValueError(
                    "matmul: Input operand 1 has a mismatch in its core "
                    "dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)"
                    " (size {n} is different from {m})".format(
                        n=self._size,
                        m=x.shape[0]
                    )
                )
        elif self.ndim == 1 and x.ndim > 1:
            if self._size == x.shape[-2]:
                out_shape = *x.shape[:-2], x.shape[-1]
                out = empty(out_shape)
                out2 = space[:reduce(operator.mul, out_shape)].reshape(
                    out_shape
                )

                if self.fragmented:
                    k = self._capacity - self._begin  # fragmentation index

                    matmul(self[self._begin:], x[..., :k, :], out)
                    out += matmul(self[:self._end], x[..., k:, :], out2)
                else:
                    if self._begin < self._end:
                        part = self[self._begin:self._end]
                    elif self._end == 0:
                        part = self[self._begin:]

                    matmul(part, x, out).view(ndarray)

                return(out)
            else:
                raise ValueError(
                    "matmul: Input operand 1 has a mismatch in its core "
                    "dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)"
                    " (size {n} is different from {m})".format(
                        n=self.shape[-2],
                        m=x.shape[0]
                    )
                )
        elif self.ndim == 2:
            if (self.shape[-1] == x.shape[-2]):
                out = empty(
                    (*x.shape[:-2], self.shape[-1], x.shape[-2])
                )

                if self.fragmented:
                    k = self._capacity - self._begin  # fragmentation index

                    matmul(self[self._begin:], x, out[..., :k, :])
                    matmul(self[:self._end], x, out[..., k:, :])
                else:
                    if self._begin < self._end:
                        part = self[self._begin:self._end]
                    elif self._end == 0:
                        part = self[self._begin:]

                    matmul(part, x, out)

                return(out.view(ndarray))

            else:
                raise ValueError(
                    (
                        "matmul: Input operand 1 has a mismatch in its core "
                        "dimension 0, with gufunc signature (n?,k),(k,m?)->"
                        "(n?,m?) (size {n} is different from {m})"
                    ).format(
                        n=self.shape[-1],
                        m=x.shape[-2]
                    )
                )
        else:
            if (self.shape[-1] == x.shape[-2]):
                self_shape = (self._size, *self.shape[1:-2])

                starexpr = tuple(
                    zip_longest(self_shape, x.shape[:-2], fillvalue=1)
                )
                if star_can_broadcast(starexpr):
                    broadcast_shape = tuple(
                        starmap(
                            lambda a, b: max(a, b),
                            starexpr
                        )
                    )

                    out = empty(
                        (*broadcast_shape, self.shape[-2], x.shape[-1])
                    )

                    if self.fragmented:
                        k = self._capacity - self._begin  # fragmentation index

                        if x.ndim > 2:
                            matmul(self[self._begin:], x[:k], out[:k])
                            matmul(self[:self._end], x[k:], out[k:])
                        else:
                            matmul(self[self._begin:], x, out[:k])
                            matmul(self[:self._end], x, out[k:])
                    else:
                        if self._begin < self._end:
                            part = self[self._begin:self._end]
                        elif self._end == 0:
                            part = self[self._begin:]

                        matmul(part, x, out)

                    return(out.view(ndarray))
                else:
                    raise ValueError(
                        (
                            "operands could not be broadcast together with"
                            "remapped shapes [original->remapped]: "
                            "{shape_b}->({shape_bn}, newaxis,newaxis) "
                            "{shape_a}->({shape_an}, newaxis,newaxis) "
                            "and requested shape ({n},{m})"
                        ).format(
                            shape_a=self_shape,
                            shape_b=x.shape,
                            shape_an=self.shape[:-2].__str__()[:-1],
                            shape_bn=x.shape[:-2].__str__()[:-1],
                            n=self.shape[-1],
                            m=x.shape[-2]
                        )
                    )
            else:
                raise ValueError(
                    (
                        "matmul: Input operand 1 has a mismatch in its core "
                        "dimension 0, with gufunc signature (n?,k),(k,m?)->"
                        "(n?,m?) (size {n} is different from {m})"
                    ).format(
                        n=self.shape[-1],
                        m=x.shape[-2]
                    )
                )

    def __imatmul__(self, x):
        raise TypeError("In-place matrix multiplication is not (yet) "
                        "supported. Use 'a = a @ b' instead of 'a @= b'")

    def __rmatmul__(self, x):
        x = asarray(x)

        if (x.ndim == 0):
            ValueError(
                "matmul: Input operand 1 does not have enough dimensions "
                "(has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) "
                "requires 1"
            )

        if x.ndim == 1 and self.ndim == 1:
            # Dot product
            if x.shape[0] == self._size:
                if self.fragmented:
                    k = self._capacity - self._begin  # fragmentation index

                    out = matmul(x[:k], self[self._begin:]).view(ndarray)
                    out += matmul(x[k:], self[:self._end]).view(ndarray)
                else:
                    if self._begin < self._end:
                        part = self[self._begin:self._end]
                    elif self._end == 0:
                        part = self[self._begin:]

                    out = matmul(x, part)

                return(out.view(ndarray))
            else:
                raise ValueError(
                    "matmul: Input operand 1 has a mismatch in its core "
                    "dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)"
                    " (size {n} is different from {m})".format(
                        n=self._size,
                        m=x.shape[0]
                    )
                )
        elif x.ndim == 1 and self.ndim > 1:
            if x.shape[0] == self.shape[-2]:
                if self.ndim == 2:
                    out = empty(self.shape[-1])

                    if self.fragmented:
                        k = self._capacity - self._begin  # fragmentation index

                        matmul(x[:k], self[self._begin:], out)
                        out += matmul(x[k:], self[:self._end]).view(ndarray)
                    else:
                        if self._begin < self._end:
                            part = self[self._begin:self._end]
                        elif self._end == 0:
                            part = self[self._begin:]

                        matmul(x, part, out)

                    return(out)
                else:
                    out = empty(
                        (self._size, *self.shape[1:-2], self.shape[-1])
                    )

                    if self.fragmented:
                        k = self._capacity - self._begin  # fragmentation index

                        matmul(x, self[self._begin:], out[:k])
                        matmul(x, self[:self._end], out[k:])
                    else:
                        if self._begin < self._end:
                            part = self[self._begin:self._end]
                        elif self._end == 0:
                            part = self[self._begin:]

                        matmul(x, part, out)

                    return(out)
            else:
                raise ValueError(
                    "matmul: Input operand 1 has a mismatch in its core "
                    "dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)"
                    " (size {n} is different from {m})".format(
                        n=self.shape[-2],
                        m=x.shape[0]
                    )
                )
        elif x.ndim > 1 and self.ndim == 1:
            if x.shape[-1] == self.shape[0]:
                out = empty(x.shape[:-1])
                if self.fragmented:
                    k = self._capacity - self._begin  # fragmentation index

                    matmul(x[..., :, :k], self[self._begin:], out)
                    out += matmul(x[..., :, k:], self[:self._end]).view(
                        ndarray
                    )
                else:
                    if self._begin < self._end:
                        part = self[self._begin:self._end]
                    elif self._end == 0:
                        part = self[self._begin:]

                    matmul(x, part, out)

                return(out)
            else:
                raise ValueError(
                    "matmul: Input operand 1 has a mismatch in its core "
                    "dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)"
                    " (size {n} is different from {m})".format(
                        n=self.shape[-2],
                        m=x.shape[0]
                    )
                )
        elif self.ndim == 2:
            if (x.shape[-1] == self.shape[-2]):
                out = empty(
                    (*x.shape[:-1], self.shape[-2])
                )

                if self.fragmented:
                    k = self._capacity - self._begin  # fragmentation index

                    matmul(x[..., :, :k], self[self._begin:], out)
                    out += matmul(x[..., :, k:], self[:self._end]).view(
                        ndarray
                    )

                else:
                    if self._begin < self._end:
                        part = self[self._begin:self._end]
                    elif self._end == 0:
                        part = self[self._begin:]

                    matmul(x, part, out)

                return(out.view(ndarray))

            else:
                raise ValueError(
                    (
                        "matmul: Input operand 1 has a mismatch in its core "
                        "dimension 0, with gufunc signature (n?,k),(k,m?)->"
                        "(n?,m?) (size {n} is different from {m})"
                    ).format(
                        n=self.shape[-1],
                        m=x.shape[-2]
                    )
                )
        else:
            if (x.shape[-1] == self.shape[-2]):
                self_shape = (self._size, *self.shape[1:-2])

                starexpr = tuple(
                    zip_longest(self_shape, x.shape[:-2], fillvalue=1)
                )
                if star_can_broadcast(starexpr):
                    broadcast_shape = tuple(
                        starmap(
                            lambda a, b: max(a, b),
                            starexpr
                        )
                    )

                    out = empty(
                        (*broadcast_shape, x.shape[-2], self.shape[-1])
                    )

                    if self.fragmented:
                        k = self._capacity - self._begin  # fragmentation index

                        if x.ndim > 2:
                            matmul(x[:k], self[self._begin:], out[:k])
                            matmul(x[k:], self[:self._end], out[k:])
                        else:
                            matmul(x, self[self._begin:], out[:k])
                            matmul(x, self[:self._end], out[k:])
                    else:
                        if self._begin < self._end:
                            part = self[self._begin:self._end]
                        elif self._end == 0:
                            part = self[self._begin:]

                        matmul(x, part, out)

                    return(out.view(ndarray))
                else:
                    raise ValueError(
                        (
                            "operands could not be broadcast together with"
                            "remapped shapes [original->remapped]: "
                            "{shape_b}->({shape_bn}, newaxis,newaxis) "
                            "{shape_a}->({shape_an}, newaxis,newaxis) "
                            "and requested shape ({n},{m})"
                        ).format(
                            shape_a=self_shape,
                            shape_b=x.shape,
                            shape_an=self.shape[:-2].__str__()[:-1],
                            shape_bn=x.shape[:-2].__str__()[:-1],
                            n=self.shape[-1],
                            m=x.shape[-2]
                        )
                    )
            else:
                raise ValueError(
                    (
                        "matmul: Input operand 1 has a mismatch in its core "
                        "dimension 0, with gufunc signature (n?,k),(k,m?)->"
                        "(n?,m?) (size {n} is different from {m})"
                    ).format(
                        n=self.shape[-1],
                        m=x.shape[-2]
                    )
                )

    def rmatmul(self, x, work_buffer) -> ndarray:
        """
        Performs x @ buffer. This operation requires extra space when
        buffer.fragmented == True and either:
            - x.ndim == 1 and buffer.ndim > 1 or
            - x.ndim > 1 and buffer.ndim == 1 or
            - buffer.ndim == 2

        :param array_like x: Array to be multiplied by.
        :param ndarray work_buffer:
            Extra preallocated space to be used. It must be the same datatype
            as the output. While the shape of the work_buffer not matter, it
            must have space for at least as many elements as the output.


        :rtype: ndarray
        """
        x = asarray(x)
        space = asarray(work_buffer).flat

        if (x.ndim == 0):
            ValueError(
                "matmul: Input operand 1 does not have enough dimensions "
                "(has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) "
                "requires 1"
            )

        if x.ndim == 1 and self.ndim == 1:
            # Dot product
            if x.shape[0] == self._size:
                if self.fragmented:
                    k = self._capacity - self._begin  # fragmentation index

                    out = matmul(x[:k], self[self._begin:]).view(ndarray)
                    out += matmul(x[k:], self[:self._end]).view(ndarray)
                else:
                    if self._begin < self._end:
                        part = self[self._begin:self._end]
                    elif self._end == 0:
                        part = self[self._begin:]

                    out = matmul(x, part)

                return(out.view(ndarray))
            else:
                raise ValueError(
                    "matmul: Input operand 1 has a mismatch in its core "
                    "dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)"
                    " (size {n} is different from {m})".format(
                        n=self._size,
                        m=x.shape[0]
                    )
                )
        elif x.ndim == 1 and self.ndim > 1:
            if x.shape[0] == self.shape[-2]:
                if self.ndim == 2:
                    out = empty(self.shape[-1])
                    out2 = space[:self.shape[-1]]

                    if self.fragmented:
                        k = self._capacity - self._begin  # fragmentation index

                        matmul(x[:k], self[self._begin:], out)
                        out += matmul(x[k:], self[:self._end], out2)
                    else:
                        if self._begin < self._end:
                            part = self[self._begin:self._end]
                        elif self._end == 0:
                            part = self[self._begin:]

                        matmul(x, part, out)

                    return(out)
                else:
                    out = empty(
                        (self._size, *self.shape[1:-2], self.shape[-1])
                    )

                    if self.fragmented:
                        k = self._capacity - self._begin  # fragmentation index

                        matmul(x, self[self._begin:], out[:k])
                        matmul(x, self[:self._end], out[k:])
                    else:
                        if self._begin < self._end:
                            part = self[self._begin:self._end]
                        elif self._end == 0:
                            part = self[self._begin:]

                        matmul(x, part, out)

                    return(out)
            else:
                raise ValueError(
                    "matmul: Input operand 1 has a mismatch in its core "
                    "dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)"
                    " (size {n} is different from {m})".format(
                        n=self.shape[-2],
                        m=x.shape[0]
                    )
                )
        elif x.ndim > 1 and self.ndim == 1:
            if x.shape[-1] == self.shape[0]:
                out = empty(x.shape[:-1])
                out2 = space[:reduce(operator.mul, x.shape[:-1])].reshape(
                    x.shape[:-1]
                )
                if self.fragmented:
                    k = self._capacity - self._begin  # fragmentation index

                    matmul(x[..., :, :k], self[self._begin:], out)
                    out += matmul(x[..., :, k:], self[:self._end], out2)
                else:
                    if self._begin < self._end:
                        part = self[self._begin:self._end]
                    elif self._end == 0:
                        part = self[self._begin:]

                    matmul(x, part, out)

                return(out)
            else:
                raise ValueError(
                    "matmul: Input operand 1 has a mismatch in its core "
                    "dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)"
                    " (size {n} is different from {m})".format(
                        n=self.shape[-2],
                        m=x.shape[0]
                    )
                )
        elif self.ndim == 2:
            if (x.shape[-1] == self.shape[-2]):
                out_shape = (*x.shape[:-1], self.shape[-2])
                out = empty(out_shape)
                out2 = space[:reduce(operator.mul, out_shape)].reshape(
                    out_shape
                )

                if self.fragmented:
                    k = self._capacity - self._begin  # fragmentation index

                    matmul(x[..., :, :k], self[self._begin:], out)
                    out += matmul(x[..., :, k:], self[:self._end], out2)

                else:
                    if self._begin < self._end:
                        part = self[self._begin:self._end]
                    elif self._end == 0:
                        part = self[self._begin:]

                    matmul(x, part, out)

                return(out.view(ndarray))

            else:
                raise ValueError(
                    (
                        "matmul: Input operand 1 has a mismatch in its core "
                        "dimension 0, with gufunc signature (n?,k),(k,m?)->"
                        "(n?,m?) (size {n} is different from {m})"
                    ).format(
                        n=self.shape[-1],
                        m=x.shape[-2]
                    )
                )
        else:
            if (x.shape[-1] == self.shape[-2]):
                self_shape = (self._size, *self.shape[1:-2])

                starexpr = tuple(
                    zip_longest(self_shape, x.shape[:-2], fillvalue=1)
                )
                if star_can_broadcast(starexpr):
                    broadcast_shape = tuple(
                        starmap(
                            lambda a, b: max(a, b),
                            starexpr
                        )
                    )

                    out = empty(
                        (*broadcast_shape, x.shape[-2], self.shape[-1])
                    )

                    if self.fragmented:
                        k = self._capacity - self._begin  # fragmentation index

                        if x.ndim > 2:
                            matmul(x[:k], self[self._begin:], out[:k])
                            matmul(x[k:], self[:self._end], out[k:])
                        else:
                            matmul(x, self[self._begin:], out[:k])
                            matmul(x, self[:self._end], out[k:])
                    else:
                        if self._begin < self._end:
                            part = self[self._begin:self._end]
                        elif self._end == 0:
                            part = self[self._begin:]

                        matmul(x, part, out)

                    return(out.view(ndarray))
                else:
                    raise ValueError(
                        (
                            "operands could not be broadcast together with"
                            "remapped shapes [original->remapped]: "
                            "{shape_b}->({shape_bn}, newaxis,newaxis) "
                            "{shape_a}->({shape_an}, newaxis,newaxis) "
                            "and requested shape ({n},{m})"
                        ).format(
                            shape_a=self_shape,
                            shape_b=x.shape,
                            shape_an=self.shape[:-2].__str__()[:-1],
                            shape_bn=x.shape[:-2].__str__()[:-1],
                            n=self.shape[-1],
                            m=x.shape[-2]
                        )
                    )
            else:
                raise ValueError(
                    (
                        "matmul: Input operand 1 has a mismatch in its core "
                        "dimension 0, with gufunc signature (n?,k),(k,m?)->"
                        "(n?,m?) (size {n} is different from {m})"
                    ).format(
                        n=self.shape[-1],
                        m=x.shape[-2]
                    )
                )

    def get_partions(self) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        """
        Gets a slice of the buffer between the beginning and end indices.
        If the buffer is fragmented, a tuple of two slices of the two
        fragments sequentially. Concatenating the slices in the order they are
        in the tuple will return a list of elements in the correct order.

        Time complexity: O(1)

        :returns: slice or tuple of slices of the array elements in order
        :rtype: Union[ndarray, Tuple[ndarray, ndarray]]
        """
        if self.fragmented:
            return (self[self._begin:], self[:self._end])
        else:
            return self[self._begin:self._end]

    def __add__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    add(self[self._begin:], x[:k], out[:k])
                    add(self[:self._end], x[k:], out[k:])
                else:
                    add(self[self._begin:], x, out[:k])
                    add(self[:self._end], x, out[k:])
            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                add(part, x, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __radd__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index

                if x.ndim >= 1:
                    add(x[:k], self[self._begin:], out[:k])
                    add(x[k:], self[:self._end], out[k:])
                else:
                    add(x, self[self._begin:], out[:k])
                    add(x, self[:self._end], out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                add(x, part, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast"
                "together with shapes {} {}".format(
                    x.shape,
                    (self._size, *self.shape[1:])
                )
            )

    def __iadd__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    add(self[self._begin:], x[:k], self[self._begin:])
                    add(self[:self._end], x[k:], self[:self._end])
                else:
                    add(self[self._begin:], x, self[self._begin:])
                    add(self[:self._end], x, self[:self._end])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                add(part, x, part)
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __sub__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    subtract(self[self._begin:], x[:k], out[:k])
                    subtract(self[:self._end], x[k:], out[k:])
                else:
                    subtract(self[self._begin:], x, out[:k])
                    subtract(self[:self._end], x, out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                subtract(part, x, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __rsub__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index

                if x.ndim >= 1:
                    subtract(x[:k], self[self._begin:], out[:k])
                    subtract(x[k:], self[:self._end], out[k:])
                else:
                    subtract(x, self[self._begin:], out[:k])
                    subtract(x, self[:self._end], out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                subtract(x, part, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast"
                "together with shapes {} {}".format(
                    x.shape,
                    (self._size, *self.shape[1:])
                )
            )

    def __isub__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    subtract(self[self._begin:], x[:k], self[self._begin:])
                    subtract(self[:self._end], x[k:], self[:self._end])
                else:
                    subtract(self[self._begin:], x, self[self._begin:])
                    subtract(self[:self._end], x, self[:self._end])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                subtract(part, x, part)
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __mul__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    multiply(self[self._begin:], x[:k], out[:k])
                    multiply(self[:self._end], x[k:], out[k:])
                else:
                    multiply(self[self._begin:], x, out[:k])
                    multiply(self[:self._end], x, out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                multiply(part, x, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __rmul__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index

                if x.ndim >= 1:
                    multiply(x[:k], self[self._begin:], out[:k])
                    multiply(x[k:], self[:self._end], out[k:])
                else:
                    multiply(x, self[self._begin:], out[:k])
                    multiply(x, self[:self._end], out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                multiply(x, part, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast"
                "together with shapes {} {}".format(
                    x.shape,
                    (self._size, *self.shape[1:])
                )
            )

    def __imul__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    multiply(self[self._begin:], x[:k], self[self._begin:])
                    multiply(self[:self._end], x[k:], self[:self._end])
                else:
                    multiply(self[self._begin:], x, self[self._begin:])
                    multiply(self[:self._end], x, self[:self._end])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                multiply(part, x, part)
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __truediv__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    divide(self[self._begin:], x[:k], out[:k])
                    divide(self[:self._end], x[k:], out[k:])
                else:
                    divide(self[self._begin:], x, out[:k])
                    divide(self[:self._end], x, out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                divide(part, x, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __rtruediv__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index

                if x.ndim >= 1:
                    divide(x[:k], self[self._begin:], out[:k])
                    divide(x[k:], self[:self._end], out[k:])
                else:
                    divide(x, self[self._begin:], out[:k])
                    divide(x, self[:self._end], out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                divide(x, part, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast"
                "together with shapes {} {}".format(
                    x.shape,
                    (self._size, *self.shape[1:])
                )
            )

    def __itruediv__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    divide(self[self._begin:], x[:k], self[self._begin:])
                    divide(self[:self._end], x[k:], self[:self._end])
                else:
                    divide(self[self._begin:], x, self[self._begin:])
                    divide(self[:self._end], x, self[:self._end])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                divide(part, x, part)
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __floordiv__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    floor_divide(self[self._begin:], x[:k], out[:k])
                    floor_divide(self[:self._end], x[k:], out[k:])
                else:
                    floor_divide(self[self._begin:], x, out[:k])
                    floor_divide(self[:self._end], x, out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                floor_divide(part, x, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __rfloordiv__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index

                if x.ndim >= 1:
                    floor_divide(x[:k], self[self._begin:], out[:k])
                    floor_divide(x[k:], self[:self._end], out[k:])
                else:
                    floor_divide(x, self[self._begin:], out[:k])
                    floor_divide(x, self[:self._end], out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                floor_divide(x, part, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast"
                "together with shapes {} {}".format(
                    x.shape,
                    (self._size, *self.shape[1:])
                )
            )

    def __ifloordiv__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    floor_divide(self[self._begin:], x[:k], self[self._begin:])
                    floor_divide(self[:self._end], x[k:], self[:self._end])
                else:
                    floor_divide(self[self._begin:], x, self[self._begin:])
                    floor_divide(self[:self._end], x, self[:self._end])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                floor_divide(part, x, part)
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __mod__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    mod(self[self._begin:], x[:k], out[:k])
                    mod(self[:self._end], x[k:], out[k:])
                else:
                    mod(self[self._begin:], x, out[:k])
                    mod(self[:self._end], x, out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                mod(part, x, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __rmod__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index

                if x.ndim >= 1:
                    mod(x[:k], self[self._begin:], out[:k])
                    mod(x[k:], self[:self._end], out[k:])
                else:
                    mod(x, self[self._begin:], out[:k])
                    mod(x, self[:self._end], out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                mod(x, part, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast"
                "together with shapes {} {}".format(
                    x.shape,
                    (self._size, *self.shape[1:])
                )
            )

    def __imod__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    mod(self[self._begin:], x[:k], self[self._begin:])
                    mod(self[:self._end], x[k:], self[:self._end])
                else:
                    mod(self[self._begin:], x, self[self._begin:])
                    mod(self[:self._end], x, self[:self._end])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                mod(part, x, part)
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __pow__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    power(self[self._begin:], x[:k], out[:k])
                    power(self[:self._end], x[k:], out[k:])
                else:
                    power(self[self._begin:], x, out[:k])
                    power(self[:self._end], x, out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                power(part, x, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __rpow__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index

                if x.ndim >= 1:
                    power(x[:k], self[self._begin:], out[:k])
                    power(x[k:], self[:self._end], out[k:])
                else:
                    power(x, self[self._begin:], out[:k])
                    power(x, self[:self._end], out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                power(x, part, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast"
                "together with shapes {} {}".format(
                    x.shape,
                    (self._size, *self.shape[1:])
                )
            )

    def __ipow__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    power(self[self._begin:], x[:k], self[self._begin:])
                    power(self[:self._end], x[k:], self[:self._end])
                else:
                    power(self[self._begin:], x, self[self._begin:])
                    power(self[:self._end], x, self[:self._end])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                power(part, x, part)
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __and__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    bitwise_and(self[self._begin:], x[:k], out[:k])
                    bitwise_and(self[:self._end], x[k:], out[k:])
                else:
                    bitwise_and(self[self._begin:], x, out[:k])
                    bitwise_and(self[:self._end], x, out[k:])
            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                bitwise_and(part, x, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __rand__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index

                if x.ndim >= 1:
                    bitwise_and(self[self._begin:], x[:k], out[:k])
                    bitwise_and(self[:self._end], x[k:], out[k:])
                else:
                    bitwise_and(self[self._begin:], x, out[:k])
                    bitwise_and(self[:self._end], x, out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                bitwise_and(x, part, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast"
                "together with shapes {} {}".format(
                    x.shape,
                    (self._size, *self.shape[1:])
                )
            )

    def __iand__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    bitwise_and(self[self._begin:], x[:k], self[self._begin:])
                    bitwise_and(self[:self._end], x[k:], self[:self._end])
                else:
                    bitwise_and(self[self._begin:], x, self[self._begin:])
                    bitwise_and(self[:self._end], x, self[:self._end])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                bitwise_and(part, x, part)
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __or__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index

                if x.ndim >= 1:
                    bitwise_or(x[:k], self[self._begin:], out[:k])
                    bitwise_or(x[k:], self[:self._end], out[k:])
                else:
                    bitwise_or(x, self[self._begin:], out[:k])
                    bitwise_or(x, self[:self._end], out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                bitwise_or(x, part, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast"
                "together with shapes {} {}".format(
                    x.shape,
                    (self._size, *self.shape[1:])
                )
            )

    def __ror__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index

                if x.ndim >= 1:
                    bitwise_or(self[self._begin:], x[:k], out[:k])
                    bitwise_or(self[:self._end], x[k:], out[k:])
                else:
                    bitwise_or(self[self._begin:], x, out[:k])
                    bitwise_or(self[:self._end], x, out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                bitwise_or(x, part, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast"
                "together with shapes {} {}".format(
                    x.shape,
                    (self._size, *self.shape[1:])
                )
            )

    def __ior__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    bitwise_or(self[self._begin:], x[:k], self[self._begin:])
                    bitwise_or(self[:self._end], x[k:], self[:self._end])
                else:
                    bitwise_or(self[self._begin:], x, self[self._begin:])
                    bitwise_or(self[:self._end], x, self[:self._end])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                bitwise_or(part, x, part)
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __xor__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index

                if x.ndim >= 1:
                    bitwise_xor(x[:k], self[self._begin:], out[:k])
                    bitwise_xor(x[k:], self[:self._end], out[k:])
                else:
                    bitwise_xor(x, self[self._begin:], out[:k])
                    bitwise_xor(x, self[:self._end], out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                bitwise_xor(x, part, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast"
                "together with shapes {} {}".format(
                    x.shape,
                    (self._size, *self.shape[1:])
                )
            )

    def __rxor__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index

                if x.ndim >= 1:
                    bitwise_xor(self[self._begin:], x[:k], out[:k])
                    bitwise_xor(self[:self._end], x[k:], out[k:])
                else:
                    bitwise_xor(self[self._begin:], x, out[:k])
                    bitwise_xor(self[:self._end], x, out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                bitwise_xor(x, part, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast"
                "together with shapes {} {}".format(
                    x.shape,
                    (self._size, *self.shape[1:])
                )
            )

    def __ixor__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    bitwise_xor(self[self._begin:], x[:k], self[self._begin:])
                    bitwise_xor(self[:self._end], x[k:], self[:self._end])
                else:
                    bitwise_xor(self[self._begin:], x, self[self._begin:])
                    bitwise_xor(self[:self._end], x, self[:self._end])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                bitwise_xor(part, x, part)
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __rshift__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    right_shift(self[self._begin:], x[:k], out[:k])
                    right_shift(self[:self._end], x[k:], out[k:])
                else:
                    right_shift(self[self._begin:], x, out[:k])
                    right_shift(self[:self._end], x, out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                right_shift(part, x, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __rrshift__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    right_shift(x[:k], self[self._begin:], out[:k])
                    right_shift(x[k:], self[:self._end], out[k:])
                else:
                    right_shift(x, self[self._begin:], out[:k])
                    right_shift(x, self[:self._end], out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                right_shift(x, part, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __irshift__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    right_shift(self[self._begin:], x[:k], self[self._begin:])
                    right_shift(self[:self._end], x[k:], self[:self._end])
                else:
                    right_shift(self[self._begin:], x, self[self._begin:])
                    right_shift(self[:self._end], x, self[:self._end])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                right_shift(part, x, part)
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __lshift__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    left_shift(self[self._begin:], x[:k], out[:k])
                    left_shift(self[:self._end], x[k:], out[k:])
                else:
                    left_shift(self[self._begin:], x, out[:k])
                    left_shift(self[:self._end], x, out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                left_shift(part, x, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __rlshift__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            out = empty(tuple(starmap(lambda a, b: max(a, b), starexpr)))

            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    left_shift(x[:k], self[self._begin:], out[:k])
                    left_shift(x[k:], self[:self._end], out[k:])
                else:
                    left_shift(x, self[self._begin:], out[:k])
                    left_shift(x, self[:self._end], out[k:])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                left_shift(x, part, out)

            return(out.view(ndarray))
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __ilshift__(self, x):
        x = asarray(x)

        self_shape = (self._size, *self.shape[1:])
        starexpr = tuple(zip_longest(self_shape, x.shape, fillvalue=1))

        if star_can_broadcast(starexpr):
            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                if x.ndim >= 1:
                    left_shift(self[self._begin:], x[:k], self[self._begin:])
                    left_shift(self[:self._end], x[k:], self[:self._end])
                else:
                    left_shift(self[self._begin:], x, self[self._begin:])
                    left_shift(self[:self._end], x, self[:self._end])

            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                left_shift(part, x, part)
        else:
            raise ValueError(
                "operands could not be broadcast "
                "together with shapes {} {}".format(
                    (self._size, *self.shape[1:]),
                    x.shape
                )
            )

    def __invert__(self):
        out = empty((self._size, *self.shape[1:]))

        if self.fragmented:
            k = self._capacity - self._begin  # fragmentation index

            invert(self[self._begin:], out[:k])
            invert(self[:self._end], out[k:])

        else:
            if self._begin < self._end:
                part = self[self._begin:self._end]
            elif self._end == 0:
                part = self[self._begin:]

            invert(part, out)

        return(out.view(ndarray))

    def __inv__(self):
        return self.__invert__()

    def __abs__(self):
        out = empty((self._size, *self.shape[1:]))

        if self.fragmented:
            k = self._capacity - self._begin  # fragmentation index

            absolute(self[self._begin:], out[:k])
            absolute(self[:self._end], out[k:])

        else:
            if self._begin < self._end:
                part = self[self._begin:self._end]
            elif self._end == 0:
                part = self[self._begin:]

            absolute(part, out)

        return(out.view(ndarray))

    def __neg__(self):
        out = empty((self._size, *self.shape[1:]))

        if self.fragmented:
            k = self._capacity - self._begin  # fragmentation index

            negative(self[self._begin:], out[:k])
            negative(self[:self._end], out[k:])

        else:
            if self._begin < self._end:
                part = self[self._begin:self._end]
            elif self._end == 0:
                part = self[self._begin:]

            negative(part, out)

        return(out.view(ndarray))

    def __pos__(self):
        out = empty((self._size, *self.shape[1:]))

        if self.fragmented:
            k = self._capacity - self._begin  # fragmentation index

            positive(self[self._begin:], out[:k])
            positive(self[:self._end], out[k:])

        else:
            if self._begin < self._end:
                part = self[self._begin:self._end]
            elif self._end == 0:
                part = self[self._begin:]

            positive(part, out)

        return(out.view(ndarray))

    @property
    def fragmented(self) -> bool:
        """
        Property that returns True if the buffer is fragmented (the
        beginning index is greater than the end index), False otherwise.

        Time complexity: O(1)

        :returns: True if buffer is fragmented, False otherwise.
        :rtype: bool
        """
        return not (
            self._begin < self._end or
            self._end == 0
        )

    @property
    def full(self) -> bool:
        """
        Property that returns True if the buffer is full, False otherwise.

        Time complexity: O(1)

        :returns: True if buffer is full, False otherwise.
        :rtype: bool

        See Also
        --------
        :func:`NumpyCircularBuffer.empty`
        """

        return (self._size == self._capacity)

    @property
    def empty(self) -> bool:
        """
        Property that returns True if the buffer is empty, False otherwise.

        Time complexity: O(1)

        :returns: True if buffer is empty, False otherwise.
        :rtype: bool

        See Also
        --------
        :func:`NumpyCircularBuffer.full`
        """

        return (self._size == 0)

    def reset(self):
        """
        Empties all elements from the buffer.


        Time complexity: O(1)

        :returns: True if buffer is empty, False otherwise.
        :rtype: bool
        """

        self._begin = 0
        self._end = 0
        self._size = 0

    def append(self, value):
        """
        Append a value to the buffer on the right. If the buffer is full, the
        buffer will advance forward (wrapping around at the ends) and overwrite
        an element.

        Time complexity: O(1)

        See Also
        --------
        :func:`NumpyCircularBuffer.pop`
        :func:`NumpyCircularBuffer.peek`
        """
        self[self._end] = value
        self._end = (self._end + 1) % self._capacity

        if self.full:
            self._begin = (self._begin + 1) % self._capacity

        else:
            self._size += 1

    def pop(self):
        """
        Gets the element at the start of the buffer and advances the start
        of the buffer by one, consuming the element returned.

        Time complexity: O(1)

        :raises: :class:`ValueError` if buffer is empty
        :returns: element at the start of the buffer

        See Also
        --------
        :func:`NumpyCircularBuffer.peek`
        :func:`NumpyCircularBuffer.append`
        """

        if not self.empty:
            i = self._begin

            self._begin = (self._begin + 1) % self._capacity
            self._size -= 1

            return (self[i])
        else:
            raise ValueError

    def peek(self):
        """
        Gets the element at the start of the buffer without advancing the
        start of the buffer.

        Time complexity: O(1)

        :raises: :class:`ValueError` if buffer is empty
        :returns: element at the start of the buffer

        See Also
        --------
        :func:`NumpyCircularBuffer.pop`
        :func:`NumpyCircularBuffer.append`

        """
        if not self.empty:
            return (self[self._begin])
        else:
            raise ValueError

    def all(self, *args, **kwargs):
        """
        Returns True if all elements evaluate to True.

        :returns: True if all elements evaluate to True, False otherwise.

        See Also
        --------
        :func:`ndarray.all`

        """
        if self.fragmented:
            return (
                np.all(self[self._begin:].view(ndarray), *args, **kwargs) and
                np.all(self[:self._end].view(ndarray), *args, **kwargs)
            )
        else:
            if self._begin < self._end:
                part = self[self._begin:self._end]
            elif self._end == 0:
                part = self[self._begin:]

            return (np.all(part.view(ndarray), *args, **kwargs))

    def any(self, *args, **kwargs):
        """
        Returns True if any elements evaluate to True.

        :returns: True if any elements evaluate to True, False otherwise.

        See Also
        --------
        :func:`ndarray.any`
        """
        if self.fragmented:
            return (
                np.any(self[self._begin:].view(ndarray), *args, **kwargs) and
                np.any(self[:self._end].view(ndarray), *args, **kwargs)
            )
        else:
            if self._begin < self._end:
                part = self[self._begin:self._end]
            elif self._end == 0:
                part = self[self._begin:]

            return (np.any(part.view(ndarray), *args, **kwargs))

    def argmax(self, *args, **kwargs):
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """

        raise NotImplementedError

    def argmin(self, *args, **kwargs):
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """

        raise NotImplementedError

    def argpartition(self, *args, **kwargs) -> NoReturn:
        """
        :raises NotImplementedError:
            This function has no plan for implementation as of this version.
        """
        raise NotImplementedError

    def argsort(self, *args, **kwargs) -> NoReturn:
        """
        :raises NotImplementedError:
            This function has no plan for implementation as of this version.
        """
        raise NotImplementedError

    def astype(self, *args, **kwargs) -> NoReturn:
        # TODO: Considered
        """
        :raises NotImplementedError:
            This function is being considered for implementation in the future
        """
        raise NotImplementedError

    def byteswap(self, inplace=False):
        """
        Swap the bytes of the array elements over the valid range of the buffer

        Toggle between low-endian and big-endian data representation by
        returning a byteswapped array, optionally swapped in-place. Arrays of
        byte-strings are not swapped. The real and imaginary parts of a complex
        number are swapped individually.

        See Also
        --------
        :func:`ndarray.byteswap`
        """
        if inplace:
            if self.fragmented:
                (self[self._begin:].view(ndarray)).byteswap(inplace)
                (self[:self._end].view(ndarray)).byteswap(inplace)
            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                (part.view(ndarray)).byteswap(inplace)

            return self.view(ndarray)
        else:
            out = empty_like(self)
            if self.fragmented:
                k = self._capacity - self._begin  # fragmentation index
                out[:k] = (self[self._begin:].view(ndarray)).byteswap(inplace)
                out[k:] = (self[:self._end].view(ndarray)).byteswap(inplace)
            else:
                if self._begin < self._end:
                    part = self[self._begin:self._end]
                elif self._end == 0:
                    part = self[self._begin:]

                out = (part.view(ndarray)).byteswap(inplace)

            return (out)

    def choose(self, choices, out=None, mode='raise'):
        # TODO: Considered
        """
        :raises NotImplementedError:
            This function is being considered for implementation in the future
        """
        raise NotImplementedError

    def clip(self, min=None, max=None, out=None, **kwargs):
        """
        Return an array whose values are limited to [min, max] over the valid
        range of the buffer. One of max or min must be given.

        See Also
        --------
        :func:`numpy.clip`
        """
        if min is None and max is None:
            raise ValueError("One of max or min must be given")
        if out is None:
            out = empty_like(self)

        if self.fragmented:
            k = self._capacity - self._begin  # fragmentation index
            if out is self:
                np.clip(
                    min, max,
                    self[self._begin:], self[self._begin:],
                    **kwargs
                )

                np.clip(
                    min, max,
                    self[:self._end], self[:self._end],
                    **kwargs
                )
            else:
                np.clip(min, max, self[self._begin:], out[:k], **kwargs)
                np.clip(min, max, self[:self._end], out[k:], **kwargs)

            return(out.view(ndarray))
        else:
            if self._begin < self._end:
                part = self[self._begin:self._end]
            elif self._end == 0:
                part = self[self._begin:]

            np.clip(min, max, part, out, **kwargs)

            return (out.view(ndarray))

    def conj(self):
        """
        Complex-conjugate all elements over the valid range of the buffer.

        See Also
        --------
        :func:`numpy.conjugate()`
        """
        out = empty((self._size, *self.shape[1:]), self.dtype)

        if self.fragmented:
            k = self._capacity - self._begin  # fragmentation index
            np.conjugate(self[self._begin:], out[:k])
            np.conjugate(self[:self._end], out[k:])
        else:
            if self._begin < self._end:
                part = self[self._begin:self._end]
            elif self._end == 0:
                part = self[self._begin:]

            np.conjugate(part, out)

        return(out.view(ndarray))

    def conjugate(self):
        """
        Complex-conjugate all elements over the valid range of the buffer.

        See Also
        --------
        :func:`numpy.conjugate()`
        """

        out = empty((self._size, *self.shape[1:]), self.dtype)

        if self.fragmented:
            k = self._capacity - self._begin  # fragmentation index
            np.conjugate(self[self._begin:], out[:k])
            np.conjugate(self[:self._end], out[k:])
        else:
            if self._begin < self._end:
                part = self[self._begin:self._end]
            elif self._end == 0:
                part = self[self._begin:]

            np.conjugate(part, out)

        return(out)

    def copy(self, order='C', defrag=False):
        """
        Return a copy of the array over the valid range of the buffer.

        See Also
        --------
        :func:`ndarray.copy`
        """
        out = empty((self._size, *self.shape[1:]), self.dtype, order)

        if self.fragmented:
            if defrag:
                k = self._capacity - self._begin  # fragmentation index
                np.copyto(out[:k], self[self._begin:], casting='no')
                np.copyto(out[k:], self[:self._end], casting='no')
            else:
                np.copyto(out, self)
        else:
            if self._begin < self._end:
                part = self[self._begin:self._end]
            elif self._end == 0:
                part = self[self._begin:]

            np.copyto(out, part, casting='no')

        return(out)

    def cumprod(self, axis=None, dtype=None, out=None) -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def cumsum(self, axis=None, dtype=None, out=None) -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def diagonal(self, offset=0, axis1=0, axis2=1) -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def dot(self, b, out=None) -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def dump(self, b, out=None) -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def dumps(self, b, out=None) -> NoReturn:  # type: ignore
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def fill(self, value):
        """
        Fill the valid region of the buffer with a scalar value.

        :param (scalar) value: All elements of *a* will be assigned this value.

        See Also
        --------
        :func:`ndarray.fill`
        """
        if self.fragmented:
            (self[self._begin:].view(ndarray)).fill(value)
            (self[:self._end].view(ndarray)).fill(value)
        else:
            if self._begin < self._end:
                part = self[self._begin:self._end]
            elif self._end == 0:
                part = self[self._begin:]

            (part.view(ndarray)).fill(value)

    def flatten(self, order='C', defrag=False):
        """
        Return a copy of the array collapsed into one dimension.

        See Also
        --------
        :func:`ndarray.flatten`
        """
        if self.fragmented:
            if defrag:
                out = empty(self.size, self.dtype, order)

                # fragmentation index
                k = np.product(self.shape[1:]) * (self._capacity - self._begin)

                out[:k] = (self[self._begin:].view(ndarray)).flat
                out[k:] = (self[:self._end].view(ndarray)).flat
            else:
                out = (self.view(ndarray)).flatten()
        else:
            if self._begin < self._end:
                part = self[self._begin:self._end]
            elif self._end == 0:
                part = self[self._begin:]

            out = (part.view(ndarray)).flatten()
        return (out)

    def getfield(dtype, offset=0) -> NoReturn:  # type: ignore
        # TODO: Considered
        """
        :raises NotImplementedError:
            This function is being considered for implementation in the future
        """

        raise NotImplementedError

    def item(self, *args) -> NoReturn:
        # TODO: Considered
        """
        :raises NotImplementedError:
            This function is being considered for implementation in the future
        """

        raise NotImplementedError

    def itemset(self, *args) -> NoReturn:
        # TODO: Consider this for proper overloading
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def max(self, axis=None, out=None, keepdims=False, initial=None,
            where=True) -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def min(self, axis=None, out=None, keepdims=False, initial=None,
            where=True) -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def newbyteorder(self, new_order='S') -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def nonzero(self) -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function has no plan for implementation as of this version.
        """
        raise NotImplementedError

    def partition(  # type: ignore
        kth, axis=-1, kind='introselect', order=None
    ) -> NoReturn:
        """
        :raises NotImplementedError:
            This function has no plan for implementation as of this version.
        """
        raise NotImplementedError

    def prod(  # type: ignore
        axis=None,  # type: ignore
        dtype=None, out=None, keepdims=False, initial=1, where=True
    ) -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def ptp(axis=None, out=None, keepdims=False) -> NoReturn:  # type: ignore
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def put(indices, values, mode='raise') -> NoReturn:  # type: ignore
        """
        :raises NotImplementedError:
            This function has no plan for implementation as of this version.
        """
        raise NotImplementedError

    def ravel(order) -> NoReturn:  # type: ignore
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def repeat(order) -> NoReturn:  # type: ignore
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def reshape(shape, order) -> NoReturn:  # type: ignore
        """
        :raises NotImplementedError:
            This function has no plan for implementation as of this version.
        """

        raise NotImplementedError

    def resize(shape, order) -> NoReturn:  # type: ignore
        """
        :raises NotImplementedError:
            This function has no plan for implementation as of this version.
        """
        raise NotImplementedError

    def round(self, decimals=0, out=None):
        """
        Return a copy of the valid region of the buffer with each element
        rounded to the given number of decimals.

        See Also
        --------
        :func:`numpy.around`
        """
        if out is None:
            if self.flags['C_CONTIGUOUS']:
                out = empty(
                    (self._size, *self.shape[1:]), dtype=self.dtype, order='C'
                )
            else:
                out = empty(
                    (self._size, *self.shape[1:]), dtype=self.dtype, order='F'
                )

        if self.fragmented:
            if out is self:
                np.around(
                    self[self._begin:].view(ndarray),
                    decimals,
                    self[self._begin:]
                )
                np.around(
                    self[:self._end].view(ndarray),
                    decimals,
                    self[:self._end]
                )
            else:
                k = self._capacity - self._begin  # fragmentation index

                np.around(self[self._begin:].view(ndarray), decimals, out[:k])
                np.around(self[:self._end].view(ndarray), decimals, out[k:])
        else:
            if self._begin < self._end:
                part = self[self._begin:self._end]
            elif self._end == 0:
                part = self[self._begin:]

            np.around(part.view(ndarray), decimals, out)

        return(out)

    def searchsorted(self, v, side='left', sorter=None) -> NoReturn:
        """
        :raises NotImplementedError:
            This function has no plan for implementation as of this version.
        """
        raise NotImplementedError

    def setfield(self, val, dtype, offset=0) -> NoReturn:
        """
        :raises NotImplementedError:
            This function has no plan for implementation as of this version.
        """
        raise NotImplementedError

    def sort(self, axis=-1, kind=None, order=None) -> NoReturn:
        """
        :raises NotImplementedError:
            This function has no plan for implementation as of this version.
        """
        raise NotImplementedError

    def squeeze(self, axis=-1) -> NoReturn:
        # TODO:  Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def std(self, axis=-1) -> NoReturn:  # type: ignore
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def sum(self, axis=-1, dtype=None, out=None, keepdims=False, initial=0,
            where=True) -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def swapaxes(self, axis1, axis2) -> NoReturn:
        # TODO: Considered
        """
        :raises NotImplementedError:
            This function is being considered for implementation in the future
        """
        raise NotImplementedError

    def take(self, indices, axis=None, out=None, mode='raise') -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def tobytes(self, order='C') -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """

        raise NotImplementedError

    def tofile(self, fid, sep="", format="%s") -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def tolist(self) -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def tostring(self, order='C') -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError

    def transpose(self, *axes) -> NoReturn:
        """
        :raises NotImplementedError:
            This function has no plan for implementation as of this version.
        """
        raise NotImplementedError

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *,
            where=True) -> NoReturn:
        # TODO: Flagged for implementation
        """
        :raises NotImplementedError:
            This function will be implemented in the future
        """
        raise NotImplementedError
