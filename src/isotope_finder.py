"""
Contains IsotopeFinder object which detects the presence of isotopes in a
spectrum using primarily location and secondarily isotope abundance ratio
"""
import itertools

import numpy as np
from data_generation import get_isotope_data
from interpeak_distance_ratio import InterPeakDistanceRatio

"""
Plan: 
For each isotope check if there is a peak within .07 amu of each isotope peak, 
if so score each peak based on how close it is i.e distance / .07 amu. The avg 
of this score can be a feature. 

Once isotopes are found based on distance / separation compute another score
based on abundance ratio. This score is harder to come up with, for starters
compute how far off each peak is from its expected abundance, weight the
difference by the expected amount then sum it all up for another score.


try using interpeak distance which is in theother branch and checking that
abundance order matches
"""


class IsotopeFinder:
    def __init__(self, path, min_peaks=2):
        """
        Initiate IF by loading in isotopic info for known elements.

        Arguments --------
        path: Path to folder with isotopic information
        min_peaks: (Optional) Minimum number of peaks an isotope must have to
        be searched for, default is 2.
        """
        self._isotopes = {}
        self._abundances = {}
        isotope_df = get_isotope_data(path)
        for row in isotope_df.iterrows():
            name = row[1]['Element']
            if name != 'hydrocarbs':
                abunds = row[1]['Isotope Frequencies']
                masses = row[1]['Isotope Masses']
                try:
                    zero_index = abunds.index(0)
                    del abunds[zero_index]
                    del masses[zero_index]
                except ValueError:
                    pass
                isotopes = [y for x, y in sorted(zip(abunds, masses),
                                                 reverse=True)]
                abundance = sorted(abunds, reverse=True)
                if len(isotopes) >= min_peaks:
                    self._isotopes[name] = isotopes
                    self._abundances[name] = abundance

    def mz_search(self, masses: np.array, intensities, thresh=.007,
                  num_required=2) -> dict:
        """
        Search a set of masses for candidate peaks for element identification.

        Return dictionary of found elements with lists of tuples of indices for
        each isotope in the order the isotopes are in self._isotopes.

        Arguments -------
        masses: array of spectrum masses
        intensities: list of spectrum mass intensities
        thresh: (Optional) int representing how close a peak must be to an
        isotope to be considered a match, default .007 Da/amu
        num_required: (Optional) number of most abundant isotopes required to
        be a match, default is 2. If 3 or greater 2 isotope elements cannot
        be found.
        """
        candidates = {}
        for key in self._isotopes.keys():
            element_isotopes = self._isotopes[key]
            abundances = self._abundances[key]
            important_isotopes_present = True
            possibles = [[] for x in range(len(element_isotopes))]
            distances = [[] for x in range(len(element_isotopes))]
            for i, isotope in enumerate(element_isotopes):
                possibles[i] = masses[(masses > isotope - thresh) &
                                   (masses < isotope + thresh)]

                if len(possibles[i]) > 0:
                    for possible in possibles[i]:
                        distances[i].append(abs(possible - isotope))
                    indices = []
                    for possible in possibles[i]:
                        indices.append(np.where(masses == possible)[0][0])
                    possibles[i] = indices
                    # assume it is possible for interference / missing peak
                    # even if one is found
                    if abundances[i] < .02:
                        possibles[i].append(-1)
                        distances[i].append(-1)
                elif i <= num_required:
                    important_isotopes_present = False
                else:
                    possibles[i] = [-1]
                    distances[i] = [-1]
            if important_isotopes_present:
                candidates[key] = [possibles, distances]

        return candidates

    def find_atomic_patterns(self, masses, intensities, thresh=.007,
                             num_required=2,
                             distance_metric=None,
                             abundance_metric=None):
        """
        Search a spectrum for isotopic abundance patterns to use to confirm the
        presence of elements or compounds.

        Returns dictionary of found elements and their corresponding isotope
        peaks. If dists is true instead returns separation from expected
        fragment / isotope.

        Arguments -------
        masses: list of masses corresponding to spectrum peaks
        intensities: list of counts corresponding to a spectrum's peaks
        thresh: (Optional) threshold value for mz_search
        num_required: (Optional) number of required isotope peaks for mz_search
        distance_metric: (Optional) function, metric used to score distance,
        must take in 2-7 distances as a list and return a single score to be
        minimized. Defaults to calc_weighted_mean_std_sum.
        abundance_metric: (Optional) function, metric used to score abundance
        ratio, must take in 2 lists of 2-7 abundances and return a single score
        by comparing them, this score will be minimized. Defaults to
        calc_vector_distance.
        """
        candidates = self.mz_search(np.array(masses), intensities,
                                    thresh=thresh, num_required=num_required)
        avg_ht = np.mean(intensities)
        elements = {}
        for key in candidates.keys():
            combs = list(itertools.product(*candidates[key][0]))
            comb_dists = list(itertools.product(*candidates[key][1]))
            dist_scores = []
            ab_scores = []
            best_index = None
            for i in range(len(combs)):
                if not self.check_abundance_order(combs[i], intensities):
                    dist_scores.append(100000)
                    ab_scores.append(100000)
                    continue

                values = self.get_reduced_values(combs[i],comb_dists[i],
                                                 self._abundances[key])
                indices, distances, abundances = values

                if distance_metric:
                    score1 = distance_metric(distances)
                else:
                    score1 = self.calc_weighted_mean_std_sum(distances)
                dist_scores.append(score1)

                peak_abundances = self.get_abundance_ratios(indices,
                                                            intensities)
                if len(indices) > 1:  # abundance isn't relevant w/ 1 peak
                    if abundance_metric:
                        score2 = abundance_metric(peak_abundances, abundances)
                    else:
                        score2 = self.calc_vector_distance(peak_abundances,
                                                           abundances)
                    ab_scores.append(score2)
                else:
                    ab_scores.append(-10)

                if not best_index:
                    if len(indices) == 1:
                        if score1 < .002:
                            best_index = 1
                    elif ab_scores[i] < .3:
                        best_index = i
                else:
                    if dist_scores[best_index] > score1:
                        if len(indices) == 1:
                            if score1 < .002:
                                best_index = 1
                        elif ab_scores[i] < .3:
                            best_index = i

            if best_index is not None:
                best_indices = list(combs[best_index])
                best_masses = []
                for index in best_indices:
                    if index != -1:
                        best_masses.append(masses[index])
                    else:
                        best_masses.append(-1)
                elements[key] = [dist_scores[best_index],
                                 ab_scores[best_index], best_masses]

        return elements

    def compare_idrs(self, indices, element, masses, best_idr):
        """
        Compares IDR of a combination of observed peaks with those of the
        known exact Isotope Masses. Takes in the previous best observed IDR and
        compares it with a new one.

        Returns the IDR of the better observed combination as well as a boolean
        indicating whether it changed.

        Arguments --------
        indices: list of indices in masses for new combination of peaks
        element: name of element being checked
        masses: list of spectrum peak masses
        best_idr: previous best combination's IDR or None
        """
        curr_masses = list(np.array(masses)[list(indices)])
        idr = InterPeakDistanceRatio(curr_masses, element)
        changed = False
        if best_idr:
            isotopes = self._isotopes[element]
            isotopes = [y for x, y in zip(indices, isotopes) if x != -1]
            known_idr = InterPeakDistanceRatio(isotopes, element)
            total_diff_curr = sum(idr.get_dists(known_idr))
            total_diff_prev = sum(best_idr.get_dists(known_idr))
            if total_diff_curr < total_diff_prev:
                best_idr = idr
                changed=True
        else:
            best_idr = idr
            changed = True

        return best_idr, changed

    @staticmethod
    def calc_vector_distance(observed, expected):
        """
        Gets distance between two equally sized lists of abundance ratios.

        Arguments -------
        observed: abundance ratios for the observed peaks
        expected: expected abundance ratios for the peaks
        """
        if len(observed) != len(expected):
            raise ValueError("Length of observed and expected must be equal.")

        dists = [(x - y)**2 for x, y in zip(observed, expected)]
        return np.sqrt(np.sum(dists))

    @staticmethod
    def calc_weighted_mean_std_sum(values, w1=1, w2=1):
        """
        Returns the weighted sum of the mean and standard deviation of the
        observed abundance ratios or distances of a set of peaks.

        Arguments -------
        values: either a list of distances or a list of abundances
        w1: (Optional) weight applied to mean of distribution
        w2: (Optional) weight applied to std of distribution
        """
        return w1 * np.mean(values) + w2 * np.std(values)

    @staticmethod
    def get_abundance_ratios(indices, intensities):
        """
        Calculates the abundance ratios of peaks based on their
        intensities.

        Returns a list of three abundance ratios.

        Arguments -------
        indices: list of indices of peaks
        intensities: list of intensity for every peak in spectrum.
        """
        counts = [intensities[x] for x in indices]
        total = np.sum(counts)
        return [x / total for x in counts]

    @staticmethod
    def get_reduced_values(indices, distances, abundances):
        """
        Some Isotopes can have missing peaks but still be considered good
        matches. This is represented by a list containing only -1 for distances
        and indices.

        Returns new lists for indices, distances, and expected abundances
        removing those for missing peaks but retaining order.

        Arguments -------
        indices: list of indices to masses and intensities for a set of peaks
        distances: list of distances between peak and known isotope mass
        """
        missing_indices = []
        inds = []
        abunds = []
        dists = []
        for i, ind in enumerate(indices):
            if ind != -1:
                inds.append(ind)
                dists.append(distances[i])
                abunds.append(abundances[i])
        return inds, dists, abunds

    @staticmethod
    def check_abundance_order(indices, intensities):
        """
        Returns True of intensities sorted by height descending match up
        with the initial passed in indices order. If so, the spectrum is
        probably a good match. Otherwise returns false

        Arguments -------
        indices: list of indices of peaks
        intensities: list of intensity for every peak in spectrum.
        """
        counts = [intensities[x] for x in indices if x != -1]
        ordered = sorted(counts, reverse=True)
        return counts == ordered
