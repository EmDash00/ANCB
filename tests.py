import unittest

from ancb import NumpyCircularBuffer
from ancb import (  # type: ignore
    star_can_broadcast, can_broadcast
)

from numpy import array_equal, allclose
from numpy import array, zeros, arange
from numpy.random import rand, randint
from numpy import fill_diagonal

from itertools import zip_longest, chain

from operator import (
    matmul, add, sub, mul, truediv, mod, floordiv, pow,
    rshift, lshift, and_, or_, xor, neg, pos, abs, inv, invert

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

        def unop_testcase(op):
            def f(self):
                data = zeros(3, dtype=int)
                test = -arange(3, dtype=int)

                buffer = NumpyCircularBuffer(data)
                buffer.append(0)
                buffer.append(-1)
                buffer.append(-2)

                self.assertTrue(array_equal(op(buffer), op(test)))  # unfrag

                buffer.append(-3)
                test -= 1

                self.assertTrue(array_equal(op(buffer), op(test)))  # fragmented

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

                self.assertTrue(array_equal(op(buffer, x), op(test, x)))
                self.assertTrue(array_equal(op(x, buffer), op(x, test)))

                buffer.append(4)
                test += 1

                self.assertTrue(array_equal(op(buffer, x), op(test, x)))
                self.assertTrue(array_equal(op(x, buffer), op(x, test)))
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

                self.assertTrue(allclose(op(buffer, x), op(test, x)))
                self.assertTrue(allclose(op(x, buffer), op(x, test)))

                buffer.append(4)
                test += 1

                self.assertTrue(allclose(op(buffer, x), op(test, x)))
                self.assertTrue(allclose(op(x, buffer), op(x, test)))
            return f

        for op in bin_operators:
            setattr(obj, 'test_{}'.format(op.__name__), binop_testcase(op))

        for op in bitbin_operators:
            setattr(obj, 'test_{}'.format(op.__name__), bitbinop_testcase(op))

        for op in un_operators:
            setattr(obj, 'test_{}'.format(op.__name__), unop_testcase(op))

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

        buffer.pop_left()
        self.assertFalse(buffer.fragmented)

        buffer.pop_left()
        self.assertFalse(buffer.fragmented)

        buffer.pop_left()
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

        buffer.pop_left()
        self.assertTrue(allclose(buffer @ C[1:], (arange(5, 7)) @ C[1:]))

        buffer.pop_left()
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

        self.assertTrue(array_equal(buffer @ A, arange(3) @ A))
        self.assertTrue(allclose(buffer @ B, arange(3) @ B))

        buffer.append(3)
        self.assertTrue(allclose(buffer @ A, arange(1, 4) @ A))
        self.assertTrue(allclose(buffer @ B, arange(1, 4) @ B))

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

        self.assertTrue(array_equal(buffer @ A, test @ A))
        self.assertTrue(allclose(buffer @ B, test @ B))

        buffer.append(arange(9, 12))
        test += 3

        self.assertTrue(array_equal(buffer @ A, test @ A))
        self.assertTrue(allclose(buffer @ B, test @ B))

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

        self.assertTrue(array_equal(buffer @ A, test @ A))
        self.assertTrue(allclose(buffer @ B, test @ B))

        buffer.append(filler + 27)
        test += 9

        self.assertTrue(array_equal(buffer @ A, test @ A))
        self.assertTrue(allclose(buffer @ B, test @ B))
        self.assertTrue(allclose(buffer @ C, test @ C))

    def test_rmatmul_1d1d(self):
        """Tests X @ buffer where X.ndim == 1 and buffer.ndim == 1"""

        data = zeros(3)
        C = rand(3)

        buffer = NumpyCircularBuffer(data)
        buffer.append(0)
        self.assertTrue(allclose(C[:1] @ buffer, C[:1] @ arange(1)))

        buffer.append(1)
        self.assertTrue(allclose(C[:2] @ buffer, C[:2] @ arange(2)))

        buffer.append(2)
        self.assertTrue(allclose(C @ buffer, C @ arange(3)))

        buffer.append(3)
        self.assertTrue(allclose(C @ buffer, C @ arange(1, 4)))

        buffer.append(4)
        self.assertTrue(allclose(C @ buffer, C @ arange(2, 5)))

        buffer.append(5)
        self.assertTrue(allclose(C @ buffer, C @ arange(3, 6)))

        buffer.append(6)
        self.assertTrue(allclose(C @ buffer, C @ arange(4, 7)))

        buffer.pop_left()
        self.assertTrue(allclose(C[1:] @ buffer, C[1:] @ arange(5, 7)))

        buffer.pop_left()
        self.assertTrue(allclose(C[2:] @ buffer, C[2:] @ arange(6, 7)))

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

        self.assertTrue(array_equal(A @ buffer, A @ array([0, 1, 2])))

        buffer.append(3)
        self.assertTrue(array_equal(A @ buffer, A @ array([1, 2, 3])))
        self.assertTrue(allclose(B @ buffer, B @ array([1, 2, 3])))
        self.assertTrue(allclose(C @ buffer, C @ array([1, 2, 3])))

        buffer.append(4)
        self.assertTrue(array_equal(A @ buffer, A @ arange(2, 5)))
        self.assertTrue(allclose(B @ buffer, B @ arange(2, 5)))
        self.assertTrue(allclose(C @ buffer, C @ arange(2, 5)))

        buffer.append(5)
        self.assertTrue(array_equal(A @ buffer, A @ arange(3, 6)))
        self.assertTrue(allclose(B @ buffer, B @ arange(3, 6)))
        self.assertTrue(allclose(C @ buffer, C @ arange(3, 6)))

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

        self.assertTrue(allclose(A @ buffer1, A @ test1))
        self.assertTrue(allclose(A @ buffer2, A @ test2))

        buffer1.append(arange(9, 12))
        buffer2.append(arange(27, 36).reshape(3, 3))
        test1 += 3
        test2 += 9

        self.assertTrue(allclose(A @ buffer1, A @ test1))
        self.assertTrue(allclose(A @ buffer2, A @ test2))

        buffer1.append(arange(12, 15))
        buffer2.append(arange(36, 45).reshape(3, 3))
        test1 += 3
        test2 += 9

        self.assertTrue(allclose(A @ buffer1, A @ test1))
        self.assertTrue(allclose(A @ buffer2, A @ test2))

        buffer1.append(arange(15, 18))
        buffer2.append(arange(45, 54).reshape(3, 3))
        test1 += 3
        test2 += 9

        self.assertTrue(allclose(A @ buffer1, A @ test1))
        self.assertTrue(allclose(A @ buffer2, A @ test2))

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

        self.assertTrue(array_equal(A @ buffer, A @ test))
        self.assertTrue(allclose(B @ buffer, B @ test))
        self.assertTrue(allclose(C @ buffer, C @ test))

        buffer.append([9, 10, 11])
        test += 3

        self.assertTrue(array_equal(A @ buffer, A @ test))
        self.assertTrue(allclose(B @ buffer, B @ test))
        self.assertTrue(allclose(C @ buffer, C @ test))

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

        self.assertTrue(array_equal(A @ buffer, A @ test))
        self.assertTrue(allclose(B @ buffer, B @ test))
        self.assertTrue(allclose(C @ buffer, C @ test))

        buffer.append(filler + 27)
        test += 9

        self.assertTrue(array_equal(A @ buffer, A @ test))
        self.assertTrue(allclose(B @ buffer, B @ test))
        self.assertTrue(allclose(C @ buffer, C @ test))

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

        self.assertEqual(buffer.pop_left(), data[0])
        self.assertEqual(buffer.pop_left(), data[1])
        self.assertEqual(buffer.pop_left(), data[2])

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

        self.assertTrue(array_equal(buffer.pop_left(), data[0]))
        self.assertTrue(array_equal(buffer.pop_left(), data[1]))
        self.assertTrue(array_equal(buffer.pop_left(), data[2]))

        self.assertTrue(buffer.empty)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
