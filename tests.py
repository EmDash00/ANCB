import unittest

from ancb import NumpyCircularBuffer
from ancb import (  # type: ignore
    star_can_broadcast, can_broadcast
)

from numpy import array_equal, allclose, shares_memory
from numpy import array, zeros, arange, ndarray, ones, empty
from numpy.random import rand, randint
from numpy import fill_diagonal, roll

from itertools import zip_longest

from operator import (
    matmul, add, sub, mul, truediv, mod, floordiv, pow,
    rshift, lshift, and_, or_, xor, neg, pos, abs, inv, invert,
    iadd, iand, ifloordiv, ilshift, imod, imul,
    ior, ipow, irshift, isub, itruediv, ixor
)


class TestBroadcastability(unittest.TestCase):
    def test_broadcastablity(self):
        x = zeros((1, 2, 3, 4, 5))
        y = zeros((1, 1, 1, 4, 5))
        z = zeros((1, 1, 1, 3, 5))
        w = zeros(1)

        self.assertTrue(can_broadcast(x.shape, y.shape))

        self.assertFalse(can_broadcast(x.shape, z.shape))
        self.assertFalse(can_broadcast(y.shape, z.shape))

        self.assertTrue(can_broadcast(x.shape, x.shape))
        self.assertTrue(can_broadcast(y.shape, y.shape))
        self.assertTrue(can_broadcast(z.shape, z.shape))
        self.assertTrue(can_broadcast(w.shape, w.shape))

        self.assertTrue(can_broadcast(x.shape, w.shape))
        self.assertTrue(can_broadcast(y.shape, w.shape))
        self.assertTrue(can_broadcast(z.shape, w.shape))

    def test_star_broadcastablity(self):
        x = zeros((1, 2, 3, 4, 5))
        y = zeros((1, 1, 1, 4, 5))
        z = zeros((1, 1, 1, 3, 5))
        w = zeros(1)

        starexpr = zip_longest(x.shape, y.shape, fillvalue=1)
        self.assertTrue(star_can_broadcast(starexpr))

        starexpr = zip_longest(x.shape, z.shape, fillvalue=1)
        self.assertFalse(star_can_broadcast(starexpr))

        starexpr = zip_longest(y.shape, z.shape, fillvalue=1)
        self.assertFalse(star_can_broadcast(starexpr))

        starexpr = zip_longest(x.shape, x.shape, fillvalue=1)
        self.assertTrue(star_can_broadcast(starexpr))

        starexpr = zip_longest(y.shape, y.shape, fillvalue=1)
        self.assertTrue(star_can_broadcast(starexpr))

        starexpr = zip_longest(z.shape, z.shape, fillvalue=1)
        self.assertTrue(star_can_broadcast(starexpr))

        starexpr = zip_longest(w.shape, w.shape, fillvalue=1)
        self.assertTrue(star_can_broadcast(starexpr))

        starexpr = zip_longest(x.shape, w.shape, fillvalue=1)
        self.assertTrue(star_can_broadcast(starexpr))

        starexpr = zip_longest(y.shape, w.shape, fillvalue=1)
        self.assertTrue(star_can_broadcast(starexpr))

        starexpr = zip_longest(y.shape, w.shape, fillvalue=1)
        self.assertTrue(star_can_broadcast(starexpr))

        starexpr = zip_longest(z.shape, w.shape, fillvalue=1)
        self.assertTrue(star_can_broadcast(starexpr))


class OperatorTestFactory(type):

    def __new__(cls, name, bases, dct):
        obj = super().__new__(cls, name, bases, dct)

        bin_operators = [
            matmul, add, sub, mul, truediv, mod, floordiv, pow
        ]

        un_operators = [neg, pos, abs, invert, inv]

        bitbin_operators = [rshift, lshift, and_, or_, xor]

        i_operators = [
            iadd, ifloordiv, imul, ipow, isub, itruediv
        ]

        bit_ioperators = [
            ilshift, irshift, ior, iand, ixor, imod
        ]

        def unop_testcase(op):
            def f(self):
                data = zeros(3, dtype=int)
                test = -arange(3, dtype=int)

                buffer = NumpyCircularBuffer(data)
                buffer.append(0)
                buffer.append(-1)
                buffer.append(-2)

                res = op(buffer)
                self.assertIsInstance(res, ndarray)
                self.assertTrue(array_equal(res, op(test)))  # unfrag

                buffer.append(-3)
                test -= 1

                res = op(buffer)
                self.assertIsInstance(res, ndarray)
                self.assertTrue(array_equal(res, op(test)))  # frag

            return f

        def bitbinop_testcase(op):
            def f(self):
                data = zeros(3, dtype=int)
                test = arange(1, 4, dtype=int)

                x = randint(3)

                buffer = NumpyCircularBuffer(data)
                buffer.append(1)
                buffer.append(2)
                buffer.append(3)

                res1 = op(buffer, x)
                res2 = op(x, buffer)
                self.assertIsInstance(res1, ndarray)
                self.assertIsInstance(res2, ndarray)

                self.assertTrue(array_equal(res1, op(test, x)))
                self.assertTrue(array_equal(res2, op(x, test)))

                buffer.append(4)
                test += 1

                res1 = op(buffer, x)
                res2 = op(x, buffer)
                self.assertIsInstance(res1, ndarray)
                self.assertIsInstance(res2, ndarray)

                self.assertTrue(array_equal(res1, op(test, x)))
                self.assertTrue(array_equal(res2, op(x, test)))
            return f

        def binop_testcase(op):
            def f(self):
                data = zeros(3, dtype=float)
                test = arange(1, 4, dtype=float)

                x = rand(3)

                buffer = NumpyCircularBuffer(data)
                buffer.append(1)
                buffer.append(2)
                buffer.append(3)

                res1 = op(buffer, x)
                self.assertIsInstance(res1, ndarray)
                self.assertTrue(allclose(res1, op(test, x)))

                res2 = op(x, buffer)
                self.assertIsInstance(res2, ndarray)
                self.assertTrue(allclose(res2, op(x, test)))

                buffer.append(4)
                test += 1

                res1 = op(buffer, x)
                self.assertIsInstance(res1, ndarray)
                self.assertTrue(allclose(res1, op(test, x)))

                res2 = op(x, buffer)
                self.assertIsInstance(res2, ndarray)
                self.assertTrue(allclose(res2, op(x, test)))
            return f

        def iop_testcase(op):
            def f(self):
                data = zeros(3, dtype=float)
                data2 = zeros(3, dtype=float)

                test1 = arange(1, 4, dtype=float)
                test2 = arange(2, 5, dtype=float)

                x = rand(3)

                buffer1 = NumpyCircularBuffer(data)
                buffer2 = NumpyCircularBuffer(data2)

                buffer1.append(1)
                buffer1.append(2)
                buffer1.append(3)

                buffer2.append(1)
                buffer2.append(2)
                buffer2.append(3)

                op(buffer1, x)
                op(test1, x)
                self.assertIsInstance(buffer1, NumpyCircularBuffer)
                self.assertTrue(array_equal(buffer1 + 0, test1))

                buffer2.append(4)

                op(buffer2, x)
                op(test2, x)
                self.assertIsInstance(buffer2, NumpyCircularBuffer)
                self.assertTrue(array_equal(buffer2 + 0, test2))

            return f

        def bitiop_testcase(op):
            def f(self):
                data = zeros(3, dtype=int)
                data2 = zeros(3, dtype=int)

                test1 = arange(1, 4, dtype=int)
                test2 = arange(2, 5, dtype=int)

                x = randint(low=1, high=100, size=3)

                buffer1 = NumpyCircularBuffer(data)
                buffer2 = NumpyCircularBuffer(data2)

                buffer1.append(1)
                buffer1.append(2)
                buffer1.append(3)

                buffer2.append(1)
                buffer2.append(2)
                buffer2.append(3)

                op(buffer1, x)
                op(test1, x)
                self.assertIsInstance(buffer1, NumpyCircularBuffer)
                self.assertTrue(allclose(buffer1 + 0, test1))

                buffer2.append(4)

                op(buffer2, x)
                op(test2, x)
                self.assertIsInstance(buffer2, NumpyCircularBuffer)
                self.assertTrue(allclose(buffer2 + 0, test2))

            return f

        for op in bin_operators:
            setattr(obj, 'test_{}'.format(op.__name__), binop_testcase(op))

        for op in bitbin_operators:
            setattr(obj, 'test_{}'.format(op.__name__), bitbinop_testcase(op))

        for op in un_operators:
            setattr(obj, 'test_{}'.format(op.__name__), unop_testcase(op))

        for op in i_operators:
            setattr(obj, 'test_{}'.format(op.__name__), iop_testcase(op))

        for op in bit_ioperators:
            setattr(obj, 'test_{}'.format(op.__name__), bitiop_testcase(op))

        return(obj)


class TestNumpyCircularBuffer(
    unittest.TestCase, metaclass=OperatorTestFactory
):
    """
    NumpyCircularBuffer tests
    """

    def test_init(self):
        data = zeros(3)
        buffer = NumpyCircularBuffer(data)
        self.assertTrue(array_equal(data, buffer))

    def test_fragmentation(self):
        data = zeros(3)

        buffer = NumpyCircularBuffer(data)
        self.assertFalse(buffer.fragmented)

        buffer.append(0)
        self.assertFalse(buffer.fragmented)

        buffer.append(1)
        self.assertFalse(buffer.fragmented)

        buffer.append(2)
        self.assertFalse(buffer.fragmented)

        buffer.append(3)
        self.assertTrue(buffer.fragmented)

        buffer.append(4)
        self.assertTrue(buffer.fragmented)

        buffer.append(5)
        self.assertFalse(buffer.fragmented)

        buffer.pop()
        self.assertFalse(buffer.fragmented)

        buffer.pop()
        self.assertFalse(buffer.fragmented)

        buffer.pop()
        self.assertFalse(buffer.fragmented)

    def test_matmul_1d1d(self):
        """Tests buffer @ X where buffer.ndim == 1 and X.ndim == 1"""

        data = zeros(3)
        C = rand(3)

        buffer = NumpyCircularBuffer(data)
        buffer.append(0)
        self.assertTrue(allclose(buffer @ C[:1], arange(1) @ C[:1]))

        buffer.append(1)
        self.assertTrue(allclose(buffer @ C[:2], arange(2) @ C[:2]))

        buffer.append(2)
        self.assertTrue(allclose(buffer @ C, arange(3) @ C))

        buffer.append(3)
        self.assertTrue(allclose(buffer @ C, (arange(1, 4)) @ C))

        buffer.append(4)
        self.assertTrue(allclose(buffer @ C, (arange(2, 5)) @ C))

        buffer.append(5)
        self.assertTrue(allclose(buffer @ C, (arange(3, 6)) @ C))

        buffer.append(6)
        self.assertTrue(allclose(buffer @ C, (arange(4, 7)) @ C))

        buffer.pop()
        self.assertTrue(allclose(buffer @ C[1:], (arange(5, 7)) @ C[1:]))

        buffer.pop()
        self.assertTrue(allclose(buffer @ C[2:], (arange(6, 7)) @ C[2:]))

    def test_matmul_1d2d(self):
        """Tests buffer @ X where buffer.ndim == 1 and X.ndim == 2"""

        data = zeros(3)
        A = zeros((3, 3))
        B = rand(9).reshape(3, 3)
        fill_diagonal(A, [1, 2, 3])

        buffer = NumpyCircularBuffer(data)
        buffer.append(0)
        buffer.append(1)
        buffer.append(2)

        res_a = buffer @ A
        res_b = buffer @ B

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)

        self.assertTrue(array_equal(res_a, arange(3) @ A))
        self.assertTrue(allclose(res_b, arange(3) @ B))

        buffer.append(3)

        res_a = buffer @ A
        res_b = buffer @ B

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)

        self.assertTrue(allclose(res_a, arange(1, 4) @ A))
        self.assertTrue(allclose(res_b, arange(1, 4) @ B))

    def test_matmul_2d2d(self):
        """Tests buffer @ X where buffer.ndim == 2"""

        data = zeros((3, 3))
        A = zeros(9).reshape(3, 3)
        B = rand(9).reshape(3, 3)

        fill_diagonal(A, arange(1, 4))
        buffer = NumpyCircularBuffer(data)

        buffer.append(arange(3))
        buffer.append(arange(3, 6))
        buffer.append(arange(6, 9))

        test = arange(9).reshape(3, 3)

        self.assertTrue(array_equal(buffer, test))

        res_a = buffer @ A
        res_b = buffer @ B

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)

        self.assertTrue(array_equal(res_a, test @ A))
        self.assertTrue(allclose(res_b, test @ B))

        buffer.append(arange(9, 12))
        test += 3

        res_a = buffer @ A
        res_b = buffer @ B

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)

        self.assertTrue(array_equal(res_a, test @ A))
        self.assertTrue(allclose(res_b, test @ B))

    def test_matmul_ndnd(self):
        """Tests buffer @ X where X.ndim > 2 and buffer.ndim > 2"""
        data = zeros((3, 3, 3))
        A = zeros((3, 3, 3))
        B = rand(27).reshape(3, 3, 3)
        C = rand(12).reshape(3, 4)

        fill_diagonal(A, [1, 2, 3])
        buffer = NumpyCircularBuffer(data)
        filler = arange(9).reshape(3, 3)

        buffer.append(filler)
        buffer.append(filler + 9)
        buffer.append(filler + 18)

        test = arange(27).reshape(3, 3, 3)

        res_a = buffer @ A
        res_b = buffer @ B

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)

        self.assertTrue(array_equal(res_a, test @ A))
        self.assertTrue(allclose(res_b, test @ B))

        buffer.append(filler + 27)
        test += 9

        res_a = buffer @ A
        res_b = buffer @ B
        res_c = buffer @ C

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, test @ A))
        self.assertTrue(allclose(res_b, test @ B))
        self.assertTrue(allclose(res_c, test @ C))

    def test_rmatmul_1d1d(self):
        """Tests X @ buffer where X.ndim == 1 and buffer.ndim == 1"""

        data = zeros(3)
        C = rand(3)

        buffer = NumpyCircularBuffer(data)

        buffer.append(0)

        res_c = C[:1] @ buffer
        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C[:1] @ arange(1)))

        buffer.append(1)

        res_c = C[:2] @ buffer
        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C[:2] @ arange(2)))

        buffer.append(2)

        res_c = C @ buffer
        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C @ arange(3)))

        buffer.append(3)

        res_c = C @ buffer
        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C @ arange(1, 4)))

        buffer.append(4)

        res_c = C @ buffer
        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C @ arange(2, 5)))

        buffer.append(5)

        res_c = C @ buffer
        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C @ arange(3, 6)))

        buffer.append(6)
        res_c = C @ buffer
        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C @ arange(4, 7)))

        buffer.pop()

        res_c = C[1:] @ buffer
        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C[1:] @ arange(5, 7)))

        buffer.pop()

        res_c = C[2:] @ buffer
        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C[2:] @ arange(6, 7)))

    def test_rmatmul_nd1d(self):
        """Tests X @ buffer where X.ndim == 1 and buffer.ndim > 1"""

        data = zeros(3)
        A = zeros(9).reshape(3, 3)
        B = arange(9).reshape(3, 3)
        C = arange(3)
        fill_diagonal(A, [1, 2, 3])

        buffer = NumpyCircularBuffer(data)
        buffer.append(0)
        buffer.append(1)
        buffer.append(2)

        res_a = A @ buffer

        self.assertIsInstance(res_a, ndarray)
        self.assertTrue(array_equal(A @ buffer, A @ array([0, 1, 2])))

        buffer.append(3)

        res_a = A @ buffer
        res_b = B @ buffer
        res_c = C @ buffer

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, A @ array([1, 2, 3])))
        self.assertTrue(allclose(res_b, B @ array([1, 2, 3])))
        self.assertTrue(allclose(res_c, C @ array([1, 2, 3])))

        buffer.append(4)

        res_a = A @ buffer
        res_b = B @ buffer
        res_c = C @ buffer

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, A @ arange(2, 5)))
        self.assertTrue(allclose(res_b, B @ arange(2, 5)))
        self.assertTrue(allclose(res_c, C @ arange(2, 5)))

        buffer.append(5)

        res_a = A @ buffer
        res_b = B @ buffer
        res_c = C @ buffer

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, A @ arange(3, 6)))
        self.assertTrue(allclose(res_b, B @ arange(3, 6)))
        self.assertTrue(allclose(res_c, C @ arange(3, 6)))

    def test_rmatmul_1dnd(self):
        """Tests X @ buffer where X.ndim == 1 and buffer.ndim > 1"""

        data1 = zeros((3, 3))
        data2 = zeros((3, 3, 3))

        A = rand(3)
        test1 = arange(9).reshape(3, 3)
        test2 = arange(27).reshape(3, 3, 3)

        buffer1 = NumpyCircularBuffer(data1)
        buffer2 = NumpyCircularBuffer(data2)

        buffer1.append(arange(3))
        buffer1.append(arange(3, 6))
        buffer1.append(arange(6, 9))

        buffer2.append(arange(9).reshape(3, 3))
        buffer2.append(arange(9, 18).reshape(3, 3))
        buffer2.append(arange(18, 27).reshape(3, 3))

        res_buf1 = A @ buffer1
        res_buf2 = A @ buffer2

        self.assertIsInstance(res_buf1, ndarray)
        self.assertIsInstance(res_buf2, ndarray)

        self.assertTrue(allclose(res_buf1, A @ test1))
        self.assertTrue(allclose(res_buf2, A @ test2))

        buffer1.append(arange(9, 12))
        buffer2.append(arange(27, 36).reshape(3, 3))
        test1 += 3
        test2 += 9

        res_buf1 = A @ buffer1
        res_buf2 = A @ buffer2

        self.assertIsInstance(res_buf1, ndarray)
        self.assertIsInstance(res_buf2, ndarray)

        self.assertTrue(allclose(res_buf1, A @ test1))
        self.assertTrue(allclose(res_buf2, A @ test2))

        buffer1.append(arange(12, 15))
        buffer2.append(arange(36, 45).reshape(3, 3))
        test1 += 3
        test2 += 9

        res_buf1 = A @ buffer1
        res_buf2 = A @ buffer2

        self.assertIsInstance(res_buf1, ndarray)
        self.assertIsInstance(res_buf2, ndarray)

        self.assertTrue(allclose(res_buf1, A @ test1))
        self.assertTrue(allclose(res_buf2, A @ test2))

        buffer1.append(arange(15, 18))
        buffer2.append(arange(45, 54).reshape(3, 3))
        test1 += 3
        test2 += 9

        res_buf1 = A @ buffer1
        res_buf2 = A @ buffer2

        self.assertIsInstance(res_buf1, ndarray)
        self.assertIsInstance(res_buf2, ndarray)

        self.assertTrue(allclose(res_buf1, A @ test1))
        self.assertTrue(allclose(res_buf2, A @ test2))

    def test_rmatmul_2d2d(self):
        data = zeros((3, 3))
        A = zeros(9).reshape(3, 3)
        B = rand(9).reshape(3, 3)
        C = rand(12).reshape(4, 3)

        fill_diagonal(A, arange(1, 4))
        buffer = NumpyCircularBuffer(data)

        buffer.append(arange(3))
        buffer.append(arange(3, 6))
        buffer.append(arange(6, 9))

        test = arange(9).reshape(3, 3)

        self.assertTrue(array_equal(buffer, test))

        res_a = A @ buffer
        res_b = B @ buffer
        res_c = C @ buffer

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, A @ test))
        self.assertTrue(allclose(res_b, B @ test))
        self.assertTrue(allclose(res_c, C @ test))

        buffer.append([9, 10, 11])
        test += 3

        res_a = A @ buffer
        res_b = B @ buffer
        res_c = C @ buffer

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, A @ test))
        self.assertTrue(allclose(res_b, B @ test))
        self.assertTrue(allclose(res_c, C @ test))

    def test_rmatmul_ndnd(self):
        data = zeros((3, 3, 3))
        A = zeros(27).reshape(3, 3, 3)
        B = arange(27).reshape(3, 3, 3)
        C = arange(3*8*3).reshape(3, 8, 3)

        fill_diagonal(A, [1, 2, 3])
        buffer = NumpyCircularBuffer(data)
        filler = arange(9).reshape(3, 3)

        buffer.append(filler)
        buffer.append(filler + 9)
        buffer.append(filler + 18)

        test = arange(27).reshape(3, 3, 3)

        res_a = A @ buffer
        res_b = B @ buffer
        res_c = C @ buffer

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, A @ test))
        self.assertTrue(allclose(res_b, B @ test))
        self.assertTrue(allclose(res_c, C @ test))

        buffer.append(filler + 27)
        test += 9

        res_a = A @ buffer
        res_b = B @ buffer
        res_c = C @ buffer

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, A @ test))
        self.assertTrue(allclose(res_b, B @ test))
        self.assertTrue(allclose(res_c, C @ test))

    def test_matmul2_1d1d(self):
        """Tests buffer @ X where buffer.ndim == 1 and X.ndim == 1"""

        data = zeros(3)
        C = rand(3)

        buffer = NumpyCircularBuffer(data)
        buffer.append(0)
        self.assertTrue(allclose(
                buffer.matmul(C[:1], empty(1)), arange(1) @ C[:1]
            )
        )

        buffer.append(1)
        self.assertTrue(allclose(
                buffer.matmul(C[:2], empty(2)), arange(2) @ C[:2]
            )
        )

        buffer.append(2)
        self.assertTrue(allclose(
                buffer.matmul(C, empty(3)), arange(3) @ C
            )
        )

        buffer.append(3)
        self.assertTrue(allclose(
                buffer.matmul(C, empty(3)), arange(1, 4) @ C
            )
        )

        buffer.append(4)
        self.assertTrue(allclose(
                buffer.matmul(C, empty(3)), arange(2, 5) @ C
            )
        )

        buffer.append(5)
        self.assertTrue(allclose(
                buffer.matmul(C, empty(3)), arange(3, 6) @ C
            )
        )

        buffer.append(6)
        self.assertTrue(allclose(
                buffer.matmul(C, empty(3)), arange(4, 7) @ C
            )
        )

        buffer.pop()
        self.assertTrue(allclose(
                buffer.matmul(C[1:], empty(2)), arange(5, 7) @ C[1:]
            )
        )

        buffer.pop()
        self.assertTrue(allclose(
                buffer.matmul(C[2:], empty(1)), arange(6, 7) @ C[2:]
            )
        )

    def test_matmul2_1d2d(self):
        """Tests buffer @ X where buffer.ndim == 1 and X.ndim == 2"""

        data = zeros(3)
        A = zeros((3, 3))
        B = rand(9).reshape(3, 3)
        fill_diagonal(A, [1, 2, 3])

        buffer = NumpyCircularBuffer(data)
        buffer.append(0)
        buffer.append(1)
        buffer.append(2)

        res_a = buffer.matmul(A, empty(3))
        res_b = buffer.matmul(B, empty(3))

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)

        self.assertTrue(array_equal(res_a, arange(3) @ A))
        self.assertTrue(allclose(res_b, arange(3) @ B))

        buffer.append(3)

        res_a = buffer.matmul(A, empty(3))
        res_b = buffer.matmul(B, empty(3))

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)

        self.assertTrue(allclose(res_a, arange(1, 4) @ A))
        self.assertTrue(allclose(res_b, arange(1, 4) @ B))

    def test_matmul2_2d2d(self):
        """Tests buffer @ X where buffer.ndim == 2"""

        data = zeros((3, 3))
        A = zeros(9).reshape(3, 3)
        B = rand(9).reshape(3, 3)

        fill_diagonal(A, arange(1, 4))
        buffer = NumpyCircularBuffer(data)

        buffer.append(arange(3))
        buffer.append(arange(3, 6))
        buffer.append(arange(6, 9))

        test = arange(9).reshape(3, 3)

        self.assertTrue(array_equal(buffer, test))

        res_a = buffer.matmul(A, empty((3, 3)))
        res_b = buffer.matmul(B, empty((3, 3)))

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)

        self.assertTrue(array_equal(res_a, test @ A))
        self.assertTrue(allclose(res_b, test @ B))

        buffer.append(arange(9, 12))
        test += 3

        res_a = buffer.matmul(A, empty((3, 3)))
        res_b = buffer.matmul(B, empty((3, 3)))

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)

        self.assertTrue(array_equal(res_a, test @ A))
        self.assertTrue(allclose(res_b, test @ B))

    def test_matmul2_ndnd(self):
        """Tests buffer @ X where X.ndim > 2 and buffer.ndim > 2"""
        data = zeros((3, 3, 3))
        A = zeros((3, 3, 3))
        B = rand(27).reshape(3, 3, 3)
        C = rand(12).reshape(3, 4)

        fill_diagonal(A, [1, 2, 3])
        buffer = NumpyCircularBuffer(data)
        filler = arange(9).reshape(3, 3)

        buffer.append(filler)
        buffer.append(filler + 9)
        buffer.append(filler + 18)

        test = arange(27).reshape(3, 3, 3)

        res_a = buffer.matmul(A, empty((3, 3, 3)))
        res_b = buffer.matmul(B, empty((3, 3, 3)))
        res_c = buffer.matmul(C, empty((3, 3, 4)))

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)

        self.assertTrue(array_equal(res_a, test @ A))
        self.assertTrue(allclose(res_b, test @ B))

        buffer.append(filler + 27)
        test += 9

        res_a = buffer.matmul(A, empty((3, 3, 3)))
        res_b = buffer.matmul(B, empty((3, 3, 3)))
        res_c = buffer.matmul(C, empty((3, 3, 4)))

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, test @ A))
        self.assertTrue(allclose(res_b, test @ B))
        self.assertTrue(allclose(res_c, test @ C))

    def test_rmatmul2_1d1d(self):
        """Tests X @ buffer where X.ndim == 1 and buffer.ndim == 1"""

        data = zeros(3)
        C = rand(3)

        buffer = NumpyCircularBuffer(data)

        buffer.append(0)

        res_c = buffer.rmatmul(C[:1], empty(1))

        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C[:1] @ arange(1)))

        buffer.append(1)

        res_c = buffer.rmatmul(C[:2], empty(1))

        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C[:2] @ arange(2)))

        buffer.append(2)

        res_c = buffer.rmatmul(C, empty(1))

        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C @ arange(3)))

        buffer.append(3)

        res_c = buffer.rmatmul(C, empty(1))

        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C @ arange(1, 4)))

        buffer.append(4)

        res_c = buffer.rmatmul(C, empty(1))

        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C @ arange(2, 5)))

        buffer.append(5)

        res_c = buffer.rmatmul(C, empty(1))

        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C @ arange(3, 6)))

        buffer.append(6)

        res_c = buffer.rmatmul(C, empty(1))

        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C @ arange(4, 7)))

        buffer.pop()

        res_c = buffer.rmatmul(C[1:], empty(1))

        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C[1:] @ arange(5, 7)))

        buffer.pop()

        res_c = buffer.rmatmul(C[2:], empty(1))

        self.assertIsInstance(res_c, ndarray)
        self.assertTrue(allclose(res_c, C[2:] @ arange(6, 7)))

    def test_rmatmul2_nd1d(self):
        """Tests X @ buffer where X.ndim == 1 and buffer.ndim > 1"""

        data = zeros(3)
        A = zeros(9).reshape(3, 3)
        B = arange(9).reshape(3, 3)
        C = arange(3)
        fill_diagonal(A, [1, 2, 3])

        buffer = NumpyCircularBuffer(data)
        buffer.append(0)
        buffer.append(1)
        buffer.append(2)

        res_a = A @ buffer
        buffer.rmatmul(A, empty(3))

        self.assertIsInstance(res_a, ndarray)
        self.assertTrue(array_equal(A @ buffer, A @ array([0, 1, 2])))

        buffer.append(3)

        res_a = buffer.rmatmul(A, empty(3))
        res_b = buffer.rmatmul(B, empty(3))
        res_c = buffer.rmatmul(C, empty(3))

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, A @ array([1, 2, 3])))
        self.assertTrue(allclose(res_b, B @ array([1, 2, 3])))
        self.assertTrue(allclose(res_c, C @ array([1, 2, 3])))

        buffer.append(4)

        res_a = buffer.rmatmul(A, empty(3))
        res_b = buffer.rmatmul(B, empty(3))
        res_c = buffer.rmatmul(C, empty(3))

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, A @ arange(2, 5)))
        self.assertTrue(allclose(res_b, B @ arange(2, 5)))
        self.assertTrue(allclose(res_c, C @ arange(2, 5)))

        buffer.append(5)

        res_a = buffer.rmatmul(A, empty(3))
        res_b = buffer.rmatmul(B, empty(3))
        res_c = buffer.rmatmul(C, empty(3))

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, A @ arange(3, 6)))
        self.assertTrue(allclose(res_b, B @ arange(3, 6)))
        self.assertTrue(allclose(res_c, C @ arange(3, 6)))

    def test_rmatmul2_1dnd(self):
        """Tests X @ buffer where X.ndim == 1 and buffer.ndim > 1"""

        data1 = zeros((3, 3))
        data2 = zeros((3, 3, 3))

        A = rand(3)
        test1 = arange(9).reshape(3, 3)
        test2 = arange(27).reshape(3, 3, 3)

        buffer1 = NumpyCircularBuffer(data1)
        buffer2 = NumpyCircularBuffer(data2)

        buffer1.append(arange(3))
        buffer1.append(arange(3, 6))
        buffer1.append(arange(6, 9))

        buffer2.append(arange(9).reshape(3, 3))
        buffer2.append(arange(9, 18).reshape(3, 3))
        buffer2.append(arange(18, 27).reshape(3, 3))

        res_buf1 = buffer1.rmatmul(A, empty(3))
        res_buf2 = buffer2.rmatmul(A, empty((3, 3)))

        self.assertIsInstance(res_buf1, ndarray)
        self.assertIsInstance(res_buf2, ndarray)

        self.assertTrue(allclose(res_buf1, A @ test1))
        self.assertTrue(allclose(res_buf2, A @ test2))

        buffer1.append(arange(9, 12))
        buffer2.append(arange(27, 36).reshape(3, 3))
        test1 += 3
        test2 += 9

        res_buf1 = buffer1.rmatmul(A, empty(3))
        res_buf2 = buffer2.rmatmul(A, empty((3, 3)))

        self.assertIsInstance(res_buf1, ndarray)
        self.assertIsInstance(res_buf2, ndarray)

        self.assertTrue(allclose(res_buf1, A @ test1))
        self.assertTrue(allclose(res_buf2, A @ test2))

        buffer1.append(arange(12, 15))
        buffer2.append(arange(36, 45).reshape(3, 3))
        test1 += 3
        test2 += 9

        res_buf1 = buffer1.rmatmul(A, empty(3))
        res_buf2 = buffer2.rmatmul(A, empty((3, 3)))

        self.assertIsInstance(res_buf1, ndarray)
        self.assertIsInstance(res_buf2, ndarray)

        self.assertTrue(allclose(res_buf1, A @ test1))
        self.assertTrue(allclose(res_buf2, A @ test2))

        buffer1.append(arange(15, 18))
        buffer2.append(arange(45, 54).reshape((3, 3)))
        test1 += 3
        test2 += 9

        res_buf1 = buffer1.rmatmul(A, empty(3))
        res_buf2 = buffer2.rmatmul(A, empty((3, 3)))

        self.assertIsInstance(res_buf1, ndarray)
        self.assertIsInstance(res_buf2, ndarray)

        self.assertTrue(allclose(res_buf1, A @ test1))
        self.assertTrue(allclose(res_buf2, A @ test2))

    def test_rmatmul2_2d2d(self):
        data = zeros((3, 3))
        A = zeros(9).reshape(3, 3)
        B = rand(9).reshape(3, 3)
        C = rand(12).reshape(4, 3)

        fill_diagonal(A, arange(1, 4))
        buffer = NumpyCircularBuffer(data)

        buffer.append(arange(3))
        buffer.append(arange(3, 6))
        buffer.append(arange(6, 9))

        test = arange(9).reshape(3, 3)

        self.assertTrue(array_equal(buffer, test))

        res_a = buffer.rmatmul(A, empty((3, 3)))
        res_b = buffer.rmatmul(B, empty((3, 3)))
        res_c = buffer.rmatmul(C, empty((4, 3)))

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, A @ test))
        self.assertTrue(allclose(res_b, B @ test))
        self.assertTrue(allclose(res_c, C @ test))

        buffer.append([9, 10, 11])
        test += 3

        res_a = buffer.rmatmul(A, empty((3, 3)))
        res_b = buffer.rmatmul(B, empty((3, 3)))
        res_c = buffer.rmatmul(C, empty((4, 3)))

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, A @ test))
        self.assertTrue(allclose(res_b, B @ test))
        self.assertTrue(allclose(res_c, C @ test))

    def test_rmatmul2_ndnd(self):
        data = zeros((3, 3, 3))
        A = zeros(27).reshape(3, 3, 3)
        B = arange(27).reshape(3, 3, 3)
        C = arange(3*8*3).reshape(3, 8, 3)

        fill_diagonal(A, [1, 2, 3])
        buffer = NumpyCircularBuffer(data)
        filler = arange(9).reshape(3, 3)

        buffer.append(filler)
        buffer.append(filler + 9)
        buffer.append(filler + 18)

        test = arange(27).reshape(3, 3, 3)

        res_a = buffer.rmatmul(A, empty((3, 3, 3)))
        res_b = buffer.rmatmul(B, empty((3, 3, 3)))
        res_c = buffer.rmatmul(C, empty((3, 8, 3)))

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, A @ test))
        self.assertTrue(allclose(res_b, B @ test))
        self.assertTrue(allclose(res_c, C @ test))

        buffer.append(filler + 27)
        test += 9

        res_a = buffer.rmatmul(A, empty((3, 3, 3)))
        res_b = buffer.rmatmul(B, empty((3, 3, 3)))
        res_c = buffer.rmatmul(C, empty((3, 8, 3)))

        self.assertIsInstance(res_a, ndarray)
        self.assertIsInstance(res_b, ndarray)
        self.assertIsInstance(res_c, ndarray)

        self.assertTrue(array_equal(res_a, A @ test))
        self.assertTrue(allclose(res_b, B @ test))
        self.assertTrue(allclose(res_c, C @ test))

    def test_forward_1d(self):
        data = zeros(3)
        test = zeros(3)

        buffer = NumpyCircularBuffer(data)

        buffer.append(1)
        test[0] = 1
        self.assertTrue(array_equal(buffer, test))

        buffer.append(2)
        test[1] = 2
        self.assertTrue(array_equal(buffer, test))

        buffer.append(3)
        test[2] = 3
        self.assertTrue(array_equal(buffer, test))

        self.assertTrue(buffer.full)

        self.assertEqual(buffer.pop(), data[0])
        self.assertEqual(buffer.pop(), data[1])
        self.assertEqual(buffer.pop(), data[2])

        self.assertTrue(buffer.empty)

    def test_forward_nd(self):
        data = zeros((3, 3, 3))

        buffer = NumpyCircularBuffer(data)
        test = zeros((3, 3, 3))

        buffer.append(1)
        test[0] = 1
        self.assertTrue(array_equal(buffer, test))

        buffer.append(2)
        test[1] = 2
        self.assertTrue(array_equal(buffer, test))

        buffer.append(3)
        test[2] = 3
        self.assertTrue(array_equal(buffer, test))

        self.assertTrue(buffer.full)

        self.assertTrue(array_equal(buffer.pop(), data[0]))
        self.assertTrue(array_equal(buffer.pop(), data[1]))
        self.assertTrue(array_equal(buffer.pop(), data[2]))

        self.assertTrue(buffer.empty)

    def test_peek(self):
        data = zeros((3, 3, 3))

        buffer = NumpyCircularBuffer(data)
        self.assertRaises(ValueError, buffer.peek)

        buffer.append(1)
        self.assertTrue(array_equal(buffer.peek(), ones((3, 3))))

        buffer.append(2)
        self.assertTrue(array_equal(buffer.peek(), ones((3, 3))))

        buffer.append(3)
        self.assertTrue(array_equal(buffer.peek(), ones((3, 3))))

        buffer.append(4)
        self.assertTrue(array_equal(buffer.peek(), ones((3, 3)) * 2))

        buffer.append(5)
        self.assertTrue(array_equal(buffer.peek(), ones((3, 3)) * 3))

        buffer.append(6)
        self.assertTrue(array_equal(buffer.peek(), ones((3, 3)) * 4))

    def test_all(self):
        data = zeros((3, 3, 3))
        buffer = NumpyCircularBuffer(data)

        buffer.append(1)
        self.assertTrue(buffer.all())

        buffer.append(1)
        self.assertTrue(buffer.all())

        buffer.append(0)
        self.assertFalse(buffer.all())

        buffer.append(1)
        self.assertFalse(buffer.all())

        buffer.append(1)
        self.assertFalse(buffer.all())

        buffer.append(2)
        self.assertTrue(buffer.all())

    def test_any(self):
        data = zeros((3, 3, 3))
        buffer = NumpyCircularBuffer(data)

        buffer.append([0, 0, 1])
        self.assertTrue(buffer.any())

        buffer.append(0)
        self.assertTrue(buffer.any())

        buffer.append(0)
        self.assertTrue(buffer.any())

        buffer.append(0)
        self.assertFalse(buffer.any())

        buffer.append(0)
        self.assertFalse(buffer.any())

        buffer.append(2)
        self.assertTrue(buffer.any())

    def test_byteswap(self):
        data = zeros(3)
        test = zeros(3)
        buffer = NumpyCircularBuffer(data)

        r = randint(100)
        buffer.append(r)
        test[0] = r

        r = randint(100)
        buffer.append(r)
        test[1] = r

        r = randint(100)
        buffer.append(r)
        test[2] = r

        res = buffer.byteswap()

        self.assertTrue(array_equal(res, test.byteswap()))
        self.assertIsInstance(res, ndarray)
        self.assertIsNot(buffer, res)

        r = randint(100)
        buffer.append(r)
        test[0] = r
        test = roll(test, -1)
        res = buffer.byteswap()

        self.assertTrue(array_equal(res, test.byteswap()))
        self.assertIsInstance(res, ndarray)
        self.assertIsNot(buffer, res)

        r = randint(100)
        buffer.append(r)
        test[0] = r
        test = roll(test, -1)
        res = buffer.byteswap()

        self.assertTrue(array_equal(res, test.byteswap()))
        self.assertIsInstance(res, ndarray)
        self.assertIsNot(buffer, res)

        r = randint(100)
        buffer.append(r)
        test[0] = r
        test = roll(test, -1)
        res = buffer.byteswap()

        self.assertTrue(array_equal(res, test.byteswap()))
        self.assertIsInstance(res, ndarray)
        self.assertIsNot(buffer, res)

        r = randint(100)
        buffer.append(r)
        res = buffer.byteswap()
        inplace_res = buffer.byteswap(inplace=True)

        self.assertTrue(array_equal(res, roll(inplace_res.view(ndarray), -1)))
        self.assertTrue(shares_memory(inplace_res, buffer))
        self.assertIsInstance(res, ndarray)
        self.assertIsInstance(inplace_res, ndarray)

    def test_clip(self):
        data = zeros(3, dtype=int)
        test = zeros(3, dtype=int)
        buffer = NumpyCircularBuffer(data)

        r = randint(100)
        buffer.append(r)
        test[0] = r

        r = randint(100)
        buffer.append(r)
        test[1] = r

        r = randint(100)
        buffer.append(r)
        test[2] = r

        res = buffer.clip(1, 10)

        self.assertTrue(allclose(res, test.clip(1, 10)))
        self.assertIsInstance(res, ndarray)

        r = randint(100)
        buffer.append(r)
        test[0] = r
        test = roll(test, -1)
        res = buffer.clip(1, 10)

        self.assertTrue(allclose(res, test.clip(1, 10)))
        self.assertIsInstance(res, ndarray)

        r = randint(100)
        buffer.append(r)
        test[0] = r
        test = roll(test, -1)
        res = buffer.clip(1, 10)

        self.assertTrue(allclose(res, test.clip(1, 10)))
        self.assertIsInstance(res, ndarray)

        r = randint(100)
        buffer.append(r)
        test[0] = r
        test = roll(test, -1)
        res = buffer.clip(1, 10)

        self.assertTrue(allclose(res, test.clip(1, 10)))
        self.assertIsInstance(res, ndarray)
        self.assertIsNot(buffer, res)

        r = randint(100)
        buffer.append(r)
        res = buffer.clip(1, 10)
        inplace_res = buffer.clip(1, 10, out=buffer)

        self.assertTrue(allclose(res, roll(inplace_res.view(ndarray), -1)))
        self.assertTrue(shares_memory(inplace_res.data, buffer.data))
        self.assertIsInstance(res, ndarray)
        self.assertIsInstance(inplace_res, ndarray)

    def test_conj(self):
        data = zeros(3, dtype=complex)
        test = zeros(3, dtype=complex)
        buffer = NumpyCircularBuffer(data)

        r = rand()
        buffer.append(r + r*1j)
        test[0] = r + r*1j

        r = rand()
        buffer.append(r + r*1j)
        test[1] = r + r*1j

        r = rand()
        buffer.append(r + r*1j)
        test[2] = r + r*1j

        res = buffer.conj()
        self.assertTrue(array_equal(res, test.conj()))
        self.assertIsInstance(res, ndarray)

        r = rand()
        buffer.append(r + r*1j)
        test[0] = r + r*1j
        test = roll(test, -1)

        res = buffer.conj()
        self.assertTrue(array_equal(res, test.conj()))
        self.assertIsInstance(res, ndarray)

        r = rand()
        buffer.append(r + r*1j)
        test[0] = r + r*1j
        test = roll(test, -1)

        res = buffer.conj()
        self.assertTrue(array_equal(res, test.conj()))

        r = rand()
        buffer.append(r + r*1j)
        test[0] = r + r*1j
        test = roll(test, -1)

        res = buffer.conj()
        self.assertTrue(array_equal(res, test.conj()))

    def test_copy(self):
        data = zeros(3)
        test = zeros(3)
        buffer = NumpyCircularBuffer(data)

        r = rand()
        buffer.append(r)
        test[0] = r

        r = rand()
        buffer.append(r)
        test[1] = r

        r = rand()
        buffer.append(r)
        test[2] = r

        res = buffer.copy()

        self.assertTrue(array_equal(res, test))
        self.assertTrue(array_equal(res, buffer))
        self.assertIsInstance(res, ndarray)
        self.assertIsNot(buffer, res)
        self.assertFalse(shares_memory(buffer, res))

        r = rand()
        buffer.append(r)
        test[0] = r
        res = buffer.copy()

        self.assertTrue(array_equal(res, test))
        self.assertIsInstance(res, ndarray)
        self.assertIsNot(buffer, res)
        self.assertFalse(shares_memory(buffer, res))

    def test_fill(self):
        data = zeros((3, 3, 3))
        buffer = NumpyCircularBuffer(data)
        test = zeros((3, 3, 3))

        buffer.append(0)
        buffer.fill(3)
        test[:1].fill(3)

        self.assertTrue(array_equal(test, buffer.view(ndarray)))

        buffer.append(0)
        buffer.fill(3)
        test[:2].fill(3)

        self.assertTrue(array_equal(test, buffer.view(ndarray)))

        buffer.append(0)
        buffer.fill(3)
        test[:3].fill(3)

        self.assertTrue(array_equal(test, buffer.view(ndarray)))

        buffer.append(1)
        buffer.fill(4)
        test.fill(4)

        self.assertTrue(array_equal(test, buffer.view(ndarray)))

    def test_flatten(self):
        data = zeros((3, 3, 3))
        buffer = NumpyCircularBuffer(data)

        buffer.append(arange(9).reshape(3, 3))
        buffer.append(arange(9, 18).reshape(3, 3))
        buffer.append(arange(18, 27).reshape(3, 3))

        res = buffer.flatten()
        self.assertTrue(array_equal(res, arange(27)))
        self.assertFalse(shares_memory(res, buffer))

        buffer.append(arange(27, 36).reshape(3, 3))

        res = buffer.flatten()
        self.assertTrue(array_equal(res, roll(arange(9, 36), 9)))
        self.assertTrue(
            array_equal(
                buffer.flatten(defrag=True),
                arange(9, 36)
            )
        )
        self.assertFalse(shares_memory(res, buffer))

    def test_round(self):
        data = zeros(3)
        test = zeros(3)
        buffer = NumpyCircularBuffer(data)

        r = rand() * 100
        buffer.append(r)
        test[0] = r

        r = rand() * 100
        buffer.append(r)
        test[1] = r

        r = rand() * 100
        buffer.append(r)
        test[2] = r

        res = buffer.round(decimals=2)

        self.assertTrue(array_equal(res, test.round(decimals=2)))
        self.assertIsInstance(res, ndarray)
        self.assertIsNot(buffer, res)

        r = rand() * 100
        buffer.append(r)
        test[0] = r
        test = roll(test, -1)
        res = buffer.round(decimals=2)

        self.assertTrue(array_equal(res, test.round(decimals=2)))
        self.assertIsInstance(res, ndarray)
        self.assertIsNot(buffer, res)

        r = rand() * 100
        buffer.append(r)
        test[0] = r
        test = roll(test, -1)
        res = buffer.round(decimals=2)

        self.assertTrue(array_equal(res, test.round(decimals=2)))
        self.assertIsInstance(res, ndarray)
        self.assertIsNot(buffer, res)

        r = rand() * 100
        buffer.append(r)
        res = buffer.round(decimals=2, out=buffer)
        inplace_res = buffer.round(decimals=2, out=buffer)

        self.assertTrue(array_equal(res, inplace_res))
        self.assertTrue(shares_memory(inplace_res, buffer))
        self.assertIsInstance(res, ndarray)
        self.assertIsInstance(inplace_res, ndarray)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
