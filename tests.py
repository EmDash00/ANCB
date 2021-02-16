import unittest
from ancb import NumpyCircularBuffer
from ancb import (  # type: ignore
    star_can_broadcast, can_broadcast
)
from numpy import array_equal, array, zeros, arange  # type: ignore
from numpy import fill_diagonal
from itertools import zip_longest


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


class TestNumpyCircularBuffer(unittest.TestCase):
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

    def test_add(self):
        data = zeros(3)
        x = arange(3)

        buffer = NumpyCircularBuffer(data)
        buffer.append(0)
        buffer.append(1)
        buffer.append(2)

        self.assertTrue(array_equal(buffer + x, x * 2))
        self.assertTrue(array_equal(x + buffer, x * 2))

        buffer.append(3)
        self.assertTrue(array_equal(buffer + (x + 1), (x + 1) * 2))
        self.assertTrue(array_equal((x + 1) + buffer, (x + 1) * 2))

        buffer.pop_left()

    def test_matmul_1d1d(self):
        """Tests buffer @ X where buffer.ndim == 1 and X.ndim == 1"""

        data = zeros(3)
        C = arange(3)

        buffer = NumpyCircularBuffer(data)
        buffer.append(0)
        self.assertTrue(array_equal(buffer @ C[:1], arange(1) @ C[:1]))

        buffer.append(1)
        self.assertTrue(array_equal(buffer @ C[:2], arange(2) @ C[:2]))

        buffer.append(2)
        self.assertTrue(array_equal(buffer @ C, arange(3) @ C))

        buffer.append(3)
        self.assertTrue(array_equal(buffer @ C, (arange(1, 4)) @ C))

        buffer.append(4)
        self.assertTrue(array_equal(buffer @ C, (arange(2, 5)) @ C))

        buffer.append(5)
        self.assertTrue(array_equal(buffer @ C, (arange(3, 6)) @ C))

        buffer.append(6)
        self.assertTrue(array_equal(buffer @ C, (arange(4, 7)) @ C))

        buffer.pop_left()
        self.assertTrue(array_equal(buffer @ C[1:], (arange(5, 7)) @ C[1:]))

        buffer.pop_left()
        self.assertTrue(array_equal(buffer @ C[2:], (arange(6, 7)) @ C[2:]))

    def test_matmul_1d2d(self):
        """Tests buffer @ X where buffer.ndim == 1 and X.ndim == 2"""

        data = zeros(3)
        A = zeros((3, 3))
        B = arange(9).reshape(3, 3)
        fill_diagonal(A, [1, 2, 3])

        buffer = NumpyCircularBuffer(data)
        buffer.append(0)
        buffer.append(1)
        buffer.append(2)

        self.assertTrue(array_equal(buffer @ A, array([0, 1, 2]) @ A))

        buffer.append(3)
        self.assertTrue(array_equal(buffer @ A, array([1, 2, 3]) @ A))
        self.assertTrue(array_equal(buffer @ B, array([1, 2, 3]) @ B))

    def test_matmul2(self):
        """Tests buffer @ X where buffer.ndim == 2"""

        data = zeros((3, 3))
        A = zeros(9).reshape(3, 3)
        B = arange(9).reshape(3, 3)

        fill_diagonal(A, [1, 2, 3])
        buffer = NumpyCircularBuffer(data)

        buffer.append([0, 1, 2])
        buffer.append([3, 4, 5])
        buffer.append([6, 7, 8])

        test = arange(9).reshape(3, 3)

        self.assertTrue(array_equal(buffer, test))

        self.assertTrue(array_equal(buffer @ A, test @ A))
        self.assertTrue(array_equal(buffer @ B, test @ B))

        buffer.append([9, 10, 11])
        test += 3

        self.assertTrue(array_equal(buffer @ A, test @ A))
        self.assertTrue(array_equal(buffer @ B, test @ B))

    def test_matmuln(self):
        """Tests buffer @ X where X.ndim > 2 and buffer.ndim > 2"""
        data = zeros((3, 3, 3))
        A = zeros((3, 3, 3))
        B = arange(27).reshape(3, 3, 3)
        C = arange(12).reshape(3, 4)

        fill_diagonal(A, [1, 2, 3])
        buffer = NumpyCircularBuffer(data)
        filler = arange(9).reshape(3, 3)

        buffer.append(filler)
        buffer.append(filler + 9)
        buffer.append(filler + 18)

        test = arange(27).reshape(3, 3, 3)

        self.assertTrue(array_equal(buffer @ A, test @ A))
        self.assertTrue(array_equal(buffer @ B, test @ B))

        buffer.append(filler + 27)
        test += 9

        self.assertTrue(array_equal(buffer @ A, test @ A))
        self.assertTrue(array_equal(buffer @ B, test @ B))
        self.assertTrue(array_equal(buffer @ C, test @ C))

    def test_rmatmul1(self):
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
        self.assertTrue(array_equal(B @ buffer, B @ array([1, 2, 3])))
        self.assertTrue(array_equal(C @ buffer, C @ array([1, 2, 3])))

    def test_rmatmul_1d1d(self):
        """Tests X @ buffer where X.ndim == 1 and buffer.ndim == 1"""

        data = zeros(3)
        C = arange(3)

        buffer = NumpyCircularBuffer(data)
        buffer.append(0)
        self.assertTrue(array_equal(C[:1] @ buffer, C[:1] @ arange(1)))

        buffer.append(1)
        self.assertTrue(array_equal(C[:2] @ buffer, C[:2] @ arange(2)))

        buffer.append(2)
        self.assertTrue(array_equal(C @ buffer, C @ arange(3)))

        buffer.append(3)
        self.assertTrue(array_equal(C @ buffer, C @ arange(1, 4)))

        buffer.append(4)
        self.assertTrue(array_equal(C @ buffer, C @ arange(2, 5)))

        buffer.append(5)
        self.assertTrue(array_equal(C @ buffer, C @ arange(3, 6)))

        buffer.append(6)
        self.assertTrue(array_equal(C @ buffer, C @ arange(4, 7)))

        buffer.pop_left()
        self.assertTrue(array_equal(C[1:] @ buffer, C[1:] @ arange(5, 7)))

        buffer.pop_left()
        self.assertTrue(array_equal(C[2:] @ buffer, C[2:] @ arange(6, 7)))

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
        self.assertTrue(array_equal(B @ buffer, B @ array([1, 2, 3])))
        self.assertTrue(array_equal(C @ buffer, C @ array([1, 2, 3])))

        buffer.append(4)
        self.assertTrue(array_equal(A @ buffer, A @ arange(2, 5)))
        self.assertTrue(array_equal(B @ buffer, B @ arange(2, 5)))
        self.assertTrue(array_equal(C @ buffer, C @ arange(2, 5)))

        buffer.append(5)
        self.assertTrue(array_equal(A @ buffer, A @ arange(3, 6)))
        self.assertTrue(array_equal(B @ buffer, B @ arange(3, 6)))
        self.assertTrue(array_equal(C @ buffer, C @ arange(3, 6)))

    def test_rmatmul1dnd(self):
        """Tests X @ buffer where X.ndim == 1 and buffer.ndim > 1"""

        data1 = zeros((3, 3))
        data2 = zeros((3, 3, 3))

        A = arange(3)
        test1 = arange(9).reshape(3, 3)
        test2 = arange(27).reshape(3, 3, 3)

        buffer1 = NumpyCircularBuffer(data1)
        buffer2 = NumpyCircularBuffer(data2)

        buffer1.append([0, 1, 2])
        buffer1.append([3, 4, 5])
        buffer1.append([6, 7, 8])

        buffer2.append(arange(9).reshape(3, 3))
        buffer2.append(arange(9, 18).reshape(3, 3))
        buffer2.append(arange(18, 27).reshape(3, 3))

        self.assertTrue(array_equal(A @ buffer1, A @ test1))
        self.assertTrue(array_equal(A @ buffer2, A @ test2))

        buffer1.append([9, 10, 11])
        buffer2.append(arange(27, 36).reshape(3, 3))
        test1 += 3
        test2 += 9

        self.assertTrue(array_equal(A @ buffer1, A @ test1))
        self.assertTrue(array_equal(A @ buffer2, A @ test2))

        buffer1.append([12, 13, 14])
        buffer2.append(arange(36, 45).reshape(3, 3))
        test1 += 3
        test2 += 9

        self.assertTrue(array_equal(A @ buffer1, A @ test1))
        self.assertTrue(array_equal(A @ buffer2, A @ test2))

        buffer1.append([15, 16, 17])
        buffer2.append(arange(45, 54).reshape(3, 3))
        test1 += 3
        test2 += 9

        self.assertTrue(array_equal(A @ buffer1, A @ test1))
        self.assertTrue(array_equal(A @ buffer2, A @ test2))

    def test_rmatmul_2d2d(self):
        data = zeros((3, 3))
        A = zeros(9).reshape(3, 3)
        B = arange(9).reshape(3, 3)

        fill_diagonal(A, [1, 2, 3])
        buffer = NumpyCircularBuffer(data)

        buffer.append([0, 1, 2])
        buffer.append([3, 4, 5])
        buffer.append([6, 7, 8])

        test = arange(9).reshape(3, 3)

        self.assertTrue(array_equal(buffer, test))

        self.assertTrue(array_equal(A @ buffer, A @ test))
        self.assertTrue(array_equal(B @ buffer, B @ test))

        buffer.append([9, 10, 11])
        test += 3

        self.assertTrue(array_equal(A @ buffer, A @ test))
        self.assertTrue(array_equal(B @ buffer, B @ test))

    def test_rmatmuln(self):
        data = zeros((3, 3, 3))
        A = zeros(27).reshape(3, 3, 3)
        B = arange(27).reshape(3, 3, 3)
        C = arange(12).reshape(4, 3)

        fill_diagonal(A, [1, 2, 3])
        buffer = NumpyCircularBuffer(data)
        filler = arange(9).reshape(3, 3)

        buffer.append(filler)
        buffer.append(filler + 9)
        buffer.append(filler + 18)

        test = arange(27).reshape(3, 3, 3)

        self.assertTrue(array_equal(A @ buffer, A @ test))
        self.assertTrue(array_equal(B @ buffer, B @ test))
        self.assertTrue(array_equal(C @ buffer, C @ test))

        buffer.append(filler + 27)
        test += 9

        self.assertTrue(array_equal(A @ buffer, A @ test))
        self.assertTrue(array_equal(B @ buffer, B @ test))
        self.assertTrue(array_equal(C @ buffer, C @ test))

    def test_forward(self):
        data = zeros(3)

        buffer = NumpyCircularBuffer(data)

        buffer.append(1)
        self.assertTrue(array_equal(buffer, array([1, 0, 0])))

        buffer.append(2)
        self.assertTrue(array_equal(buffer, array([1, 2, 0])))

        buffer.append(3)
        self.assertTrue(array_equal(buffer, array([1, 2, 3])))

        self.assertTrue(buffer.full)

        self.assertEqual(buffer.pop_left(), data[0])
        self.assertEqual(buffer.pop_left(), data[1])
        self.assertEqual(buffer.pop_left(), data[2])

        self.assertTrue(buffer.empty)

    def test_forward_nd(self):
        data = zeros((3, 3, 3))

        buffer = NumpyCircularBuffer(data)

        buffer.append(1)
        self.assertTrue(
            array_equal(
                buffer,
                array(
                    [
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                    ]
                )
            )
        )

        buffer.append(2)
        self.assertTrue(
            array_equal(
                buffer,
                array(
                    [
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                    ]
                )
            )
        )

        buffer.append(3)
        self.assertTrue(
            array_equal(
                buffer,
                array(
                    [
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
                    ]
                )
            )
        )

        self.assertTrue(buffer.full)

        self.assertTrue(array_equal(buffer.pop_left(), data[0]))
        self.assertTrue(array_equal(buffer.pop_left(), data[1]))
        self.assertTrue(array_equal(buffer.pop_left(), data[2]))

        self.assertTrue(buffer.empty)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
