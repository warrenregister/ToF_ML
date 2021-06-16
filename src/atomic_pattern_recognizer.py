"""
Atomic Pattern Recognizer finds isotopes of elements and verifies them by
checking isotopic abundance ratios.
"""
import numpy as np
from data_generation import get_isotope_data


class AtomicPatternRecognizer:
    def __init__(self, path):
        """
        Initiate APR by loading in isotopic abundances for known elements.

        Arguments --------
        path: Path to folder with isotopic information
        """
        self._exact_isotopes = {}
        self._abundance = {}
        self._isotopes = {}
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
                isotopes = [y for x, y in sorted(zip(abunds,
                                                    masses),
                                                reverse=True)]
                abundance = sorted(abunds, reverse=True)
                self._exact_isotopes[name] = isotopes
                self._abundance[name] = abundance
                rounded_isotopes = self.round_mz_ratio(isotopes)
                self._isotopes[name] = rounded_isotopes

        '''
        for csv in listdir(path):
            with open(path + csv) as file:
                lines = file.readlines()
                isotopes = []
                abundance = []
                name = csv.split('.')[0]
                for line in lines:
                    contents = line.split(',')

                    try:
                        isotopes.append(float(contents[0]))
                        abundance.append(float(contents[1]))
                    except ValueError:
                        pass

                self._isotopes[name] = isotopes
                self._abundance[name] = abundance
        '''

    @staticmethod
    def round_mz_ratio(masses, nums=(0, .5, 1)):
        """
        Round decimal of m/z numbers to nearest value in num. The assumption
        here is that we want more numbers to be rounded to whole numbers as
        we have fairly strong methods of eradicating bad isotope candidates and
        therefore we want more candidates rather than less to help deal with
        calibration errors.

        Returns np.array of rounded masses.

        Arguments -------
        masses: iterable of mass values for a spectra
        nums: list of decimals to round to, default (0, .5, 1)
        """
        rounded_masses = []
        nums = np.array(nums)
        for mass in masses:
            integer = int(mass)
            differences = abs(nums - (mass - integer))
            ind = np.where(differences == min(differences))
            rounded_masses.append(float(integer + nums[ind[0]]))

        return np.array(rounded_masses)

    def mz_search(self, masses: np.array, intens: list) -> dict:
        """
        Search a set of masses for candidate peaks for element identification.

        Return dictionary of found elements with lists of tuples of indices for
        each isotope in the order the isotopes are in self._isotopes.

        Arguments -------
        masses: rounded masses to make candidate finding broader
        thresh: int representing how much abundance an isotope needs to be
        considered essential for detecting an isotopic abundance pattern,
        default value is 5.
        """
        candidates = {}
        for key in self._isotopes.keys():
            isotopes = self._isotopes[key]
            if len(isotopes) <= 1:
                continue
            abundance = self._abundance[key]
            possibles = [[] for x in range(len(isotopes))]
            important_isotopes_present = True
            cands = {}
            for i, isotope in enumerate(isotopes):
                possibles[i] = list(np.where(masses == isotope)[0])
                # Check if peaks are possible matches with most abundant peaks
                if i != 0 and len(possibles[i]) != 0:
                    ab1 = abundance[0]
                    ab2 = abundance[i]
                    # currently dont handle 0 abundance
                    if ab1 != 0 and ab2 != 0:
                        #indices = possibles[i]
                        #possibles[i] = []
                        for peak in possibles[0]:
                            h1 = intens[peak]
                            est_h = (h1 / (ab1 + .001)) * ab2
                            for ind in possibles[i]:
                                def thresh_func(x):
                                    return (.01 - .5) * x + .5
                                thresh = thresh_func(ab2)
                                if abs(intens[ind] - est_h) / est_h < thresh:
                                    """
                                    if ind not in possibles[i]:
                                        possibles[i].append(ind)
                                    """
                                    added = False
                                    for cand in cands[peak]:
                                        if len(cand) == i:
                                            cand.append(ind)
                                            added = True
                                    if not added:
                                        newcands = []
                                        for cand in cands[peak]:
                                            newcand = cand[:-1].copy()
                                            newcand.append(ind)
                                            newcands.append(newcand)
                                        for cand in newcands:
                                            cands[peak].append(cand)
                elif i == 0:
                    for ind in possibles[i]:
                        cands[ind] = [[ind]]


                if len(possibles[i]) == 0:
                    if i == 0:
                        important_isotopes_present = False
                    else:  # Check if peak finder possibly missed peak
                        n = i - 1
                        while n >= 0:
                            ab1 = abundance[n]
                            ab2 = abundance[i]
                            indices = possibles[n]
                            possibles[n] = []
                            for ind in indices:
                                h1 = intens[ind]
                                estimated_height = (h1 / (ab1 + .001)) * ab2
                                minimum_height = min(intens)
                                if estimated_height < minimum_height:
                                    if ind not in possibles[n]:
                                        possibles[n].append(ind)
                            if len(possibles[n]) == 0:
                                important_isotopes_present = False
                            n -= 1

            if important_isotopes_present:
                """
                candidate = []
                for iso_cands in possibles:
                    if len(iso_cands) > 0:
                        candidate.append(iso_cands)
                    else:
                        candidate.append([-1])
                candidates[key] = list(itertools.product(*candidate))
                """
                candidate = []
                for string in cands.keys():
                    for cand in cands[string]:
                        if len(cand) == len(isotopes):
                            candidate.append(tuple(cand))
                candidates[key] = candidate

        return candidates

    def find_atomic_patterns(self, masses, intensities, thresh=.05,
                             nums=(0, .5, 1), dists=False):
        """
        Search a spectrum for isotopic abundance patterns to use to confirm the
        presence of elements or compounds.

        Returns dictionary of found elements and their corresponding isotope
        peaks. If dists is true instead returns separation from expected
        fragment / isotope.

        Arguments -------
        masses: list of masses corresponding to spectrum peaks
        intensities: list of counts corresponding to a spectrum's peaks
        thresh: (Optional) threshold value for check_abundance_match
        nums: (Optional) nums value for round_mz_ratio
        dists: (Optional) Boolean, if True returns distance from isotope instead
        of peak location.
        """
        rounded_masses = self.round_mz_ratio(masses, nums)
        candidates = self.mz_search(rounded_masses, intensities)
        elements = {}

        for key in candidates.keys():
            candidate_combs = candidates[key]
            abundances = self._abundance[key]
            matches = []
            for comb in candidate_combs:
                if len(comb) == 0:
                    continue
                ratios = self.get_abundance_ratios(comb, intensities)
                matched = self.check_abundance_order(comb, intensities)
                _, diffs = self.check_abundance_match(ratios, abundances,
                                                      thresh)

                if matched:
                    matches.append((comb, diffs))
            if len(matches) == 1:
                if self.check_if_inline(matches[0][0], masses,
                                        self._exact_isotopes[key]):
                    peaks = []
                    for ind in matches[0][0]:
                        if ind != -1:
                            peaks.append(masses[ind])
                        else:
                            peaks.append('possible')
                    elements[key] = peaks
            elif len(matches) > 1:
                j = 0
                min_max_diff = 100
                for i, match in enumerate(matches):
                    maximum = max(match[1])
                    if maximum < min_max_diff:
                        j = i
                        min_max_diff = maximum
                if self.check_if_inline(matches[0][0], masses,
                                        self._exact_isotopes[key]):
                    peaks = []
                    for ind in matches[j][0]:
                        peaks.append(masses[ind])
                    elements[key] = peaks

        if dists:
            for key in elements.keys():
                dist_list = []
                for i, peak in enumerate(elements[key]):
                    if type(peak) != str:
                        dist_list.append(self._exact_isotopes[key][i] - peak)
                    else:
                        dist_list.append(peak)
                elements[key] = dist_list

        return elements

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
        counts = [intensities[x] for x in indices]
        ordered = sorted(counts, reverse=True)
        return counts == ordered

    @staticmethod
    def check_abundance_match(ratios, true_ratios, thresh=.1):
        """
        Checks if a set of abundance ratios which could be an element are a true
        match for that elements known abundance ratio. Lengths of ratios and
        true ratios must match.

        Returns True if ratios match False otherwise, also returns list of
        differences between expected ratio and actual ratio.

        Arguments -------
        ratios: abundance ratios for peaks found in spectrum
        true_ratios: known abundance ratio for element being searched for.
        thresh: deviation from true-ratio beyond which ratios are labeled
        mismatches
        """
        if len(ratios) != len(true_ratios):
            print('Ratios length mismatch, returning False.')
            return False

        diffs = [abs(true_ratios[x] - ratios[x]) for x in range(len(ratios))]
        if max(diffs) > thresh:
            return False, diffs
        return True, diffs

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
    def check_if_inline(inds, peaks, masses, thresh=.04, thresh2 = .02):
        """
        Checks if a set of peaks considered to be the isotopes of an
        element could be plotted on a line together. If they are roughly
        in line they are likely associated with each other, if not they
        are likely random peaks.

        Arguments -------
        inds: list of relevant indices for candidate isotopes
        peaks: list of a spectrum's peaks masses
        masses: list of isotopes exact masses
        """
        if len(inds) != len(masses):
            print('Lists length mismatch, returning False.')
            return False
        diffs = [peaks[inds[i]]- masses[i] for i in range(len(masses))]
        actual_thresh = thresh
        if len(inds) == 2:
            actual_thresh = thresh2
        if np.std(diffs) < actual_thresh:
            return True
        return False