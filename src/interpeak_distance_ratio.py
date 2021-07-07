"""
Class to handle calculating and storing IDRs for various elements.
"""
from functools import total_ordering


@total_ordering
class InterPeakDistanceRatio:
    """
    Class to handle calculating and storing IDRs for various elements.
    """

    def __init__(self, nums, element_name):
        """
        Converts list of mass values to inter-peak distance ratio.
        """
        self.element_name = element_name
        nums = sorted(nums)
        self.masses = nums
        self._idr = []
        if len(nums) == 1:
            self._idr = [0]
        for i, num in enumerate(nums[:-1]):
            self._idr.append(nums[i + 1] - num)

    def get_idr(self):
        """
        Returns the idr
        """
        return self._idr


    def __repr__(self):
        """
        Returns string representation of IDR
        """
        return str(self._idr)

    def __eq__(self, other):
        """
        Returns true if object's IDR is the same.
        """
        dists = self.get_dists(other)
        return sum(dists) == 0

    def __ne__(self, other):
        """
        Returns true if object's IDR is not the same.
        """
        dists = self.get_dists(other)
        return sum(dists) != 0

    def __lt__(self, other):
        """
        Returns true if object's IDR is less than this object's.
        """
        dists = self.get_dists(other)
        return sum(dists) > 0

    @staticmethod
    def reduce_peaks(more, less):
        """
        If 2 IDRs have differing numbers of peaks but are from the same
        element we still need to compare them. This function finds which peaks
        are missing from the smaller of 2 IDR objects.
        """
        missing_inds = []
        for i, peak in enumerate(less.masses):
            if abs(more.masses()[i] - peak) > .01:
                missing_ind = i
                break

    def get_dists(self, other, abs_val=True):
        """
        Return differences between each place in two IDRS for the same
        element.

        Arguments -------
        other: Another IDR object for the same element with similar numbers of
        peaks
        abs_val: (Optional) If True distances are absolute values, default True
        """
        if self.element_name != other.element_name:
            raise ValueError("IDRS must be from the same element")

        if abs_val:
            dists = [abs(x - y) for x, y in zip(self._idr, other.get_idr())]
        else:
            dists = [x - y for x, y in zip(self._idr, other.get_idr())]
        return dists
