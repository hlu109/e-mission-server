from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
# Standard imports
from future import standard_library

standard_library.install_aliases()
from builtins import range
from builtins import *
from builtins import object
from past.utils import old_div
import logging
import math
import numpy as np
import pandas as pd
import copy
from sklearn import metrics
from numpy.linalg import norm

# our imports
import emission.storage.decorations.analysis_timeseries_queries as esda
import emission.core.common as ecc
from emission.analysis.modelling.tour_model.similarity import within_radius, filter_too_short


class Similarity(object):
    """
        NOTE: ONLY THE FOLLOWING METHODS HAVE BEEN REFACTORED: 
            __init__(), set_state(), fit(), bin_data(), bin_helper(), 
            group_trips(), match(), distance_helper()
        Other methods may break if invoked. 

        This class organizes data into bins by similarity. It then orders the 
        bins by largest to smallest and removes the bottom portion of the bins. 

        Clusters are first formed for start points and end points. Start points 
        are in the same cluster if they are within a certain number of meters 
        of each others (and the same applies for end points). Trip clusters are 
        then found by taking all the unique combinations of start/end clusters. 
        (In theory, this should yield the same trip clusters as the old 
        similarity code.) We want to keep the extra information about start/end 
        clusters to see if those are useful predictors. 

        This class has also been refactored to take an input of a dataframe of 
        trips (rather than a list)

        Args:
            data_df (DataFrame): DataFrame containing the columns 'start_lat', 
                'start_lon', 'end_lat', and 'end_lon' 
            radius (int/float): the maximum distance between any two points in 
                a bin, in meters.

        Attributes:
            data_df (DataFrame): the original DataFrame, potentially with the 
                columns 'start_bin', 'end_bin', 'trip_bin' added depending on 
                what other methods have been called on the class. (This is 
                basically the only relevant attribute as it contains all the 
                information anyone should need)
    """

    def __init__(self,
                 data_df,
                 radius_start,
                 radius_end,
                 shouldFilter=False,
                 cutoff=False):
        # if not data_df:
        #     self.data_df = pd.DataFrame()

        # In order to retrofit multiple invocation options without undertaking
        # a full restructuring, we will use the following structure for the
        # data
        # self.data will always represent the current state of the trips
        # self.bins will always represent the current state of the bins
        # In addition, self.all_data will represent all input trips
        # In addition, self.filtered_data will represent the output of "filter_too_short"
        # In addition, self.data_above_cutoff will represent the output of "delete_bins"
        # so even after finishing all three steps, we will have access to the
        # original input data
        # since we are only copying the lists (copy), not the objects in the
        # lists (deepcopy), this should not involve an undue user burden
        # I really don't see why we need to work with indices here, but in the absence of unit tests,
        # I am not comfortable with restructuring further
        self.data_df = data_df.reset_index(drop=True)
        self.data_df[['start_bin', 'end_bin', 'trip_bin']] = np.nan
        # self.set_state(self.all_data_df)
        self.start_bins = []
        self.end_bins = []
        self.trip_bins = []
        self.radius_start = float(radius_start)
        self.radius_end = float(radius_end)
        self.shouldFilter = shouldFilter
        self.cutoff = cutoff

    def set_state(self, in_data_df):
        """
        Encapsulates all the state related to this object
        so that we don't forget to update everything

        Somewhat redundant now that we're working with dataframes.
        """
        self.data_df = copy.copy(in_data_df)
        self.size = len(self.data_df)

    def fit(self):
        # if self.shouldFilter:
        #     self.filter_trips()
        self.bin_data()
        # adds the following columns to data_df: '

        # if self.cutoff:
        #     self.delete_bins()

        # deprecated since we now work with dataframes, so getting labels is
        # much easier :)
        # self.labels_ = self.get_result_labels_from_list()

    # Pull out the trip filtration code so that we can invoke the code in
    # multiple ways (with and without filteration) depending on whether we want
    # to focus on common trips or auto-labeling
    def filter_trips(self):
        self.filtered_data = filter_too_short(self.all_data, self.radius)
        self.set_state(self.filtered_data)

    #create bins
    def bin_data(self):
        self.bin_helper()  # creates bins for start points and end points
        self.group_trips()

    def calc_cutoff_bins(self):
        if len(self.bins) <= 1:
            print(f"{len(self.bins)}, no cutoff")
            self.newdata = self.data
            self.data_above_cutoff = self.newdata
            self.set_state(self.newdata)
            self.num = len(self.bins)
            return
        num = self.elbow_distance()
        logging.debug("bins = %s, elbow distance = %s" % (self.bins, num))
        sum = 0
        for i in range(len(self.bins)):
            sum += len(self.bins[i])
            if len(self.bins[i]) <= len(self.bins[num]):
                logging.debug(
                    "found weird condition, self.bins[i] = %s, self.bins[num] = %s"
                    % (self.bins[i], self.bins[num]))
                sum -= len(self.bins[i])
                num = i
                break
        logging.debug('the new number of trips is %d' % sum)
        logging.debug('the cutoff point is %d' % num)
        self.num = num

    #delete lower portion of bins
    def delete_bins(self):
        below_cutoff = []
        self.calc_cutoff_bins()
        for i in range(len(self.bins) - self.num):
            below_cutoff.append(self.bins.pop())
        newdata = []
        for bin in self.bins:
            for b in bin:
                d = self.data[b]
                newdata.append(self.data[b])
        self.newdata = newdata if len(newdata) > 1 else copy.copy(self.data)
        self.data_above_cutoff = self.newdata
        self.set_state(self.newdata)
        self.below_cutoff = below_cutoff
        self.below_cutoff.sort(key=lambda bin: len(bin), reverse=True)

    def get_result_labels_from_list(self):
        """
        Return "labels" for the trips, to be consistent with sklearn
        implementations.  This is not otherwise compatible with sklearn, but
        would be great to submit it as an example of an absolute radius, even
        if it is computationally expensive.

        It would certainly help with:
        https://stackoverflow.com/questions/48217127/distance-based-classification
        and
        https://stackoverflow.com/questions/35971441/how-to-adjust-this-dbscan-algorithm-python

        This function sets labels based on the various states for the trips.
        Pulling this out
        as a separate function to write unit tests. This would be normally be
        trivial - we would just index the all_trip_df on the trips in each bin and
        set a unique number. However, if we have filtered before binning, then the
        trip indices in the bin are the filtered trip indices, which are different
        from all_trips indices. So we need to remap from one set of indices to
        another before we can assign the labels.
        param: all_trip_df: dataframe of all trips
        param: filtered_trip_df: dataframe of trips that were removed as "too short"
        param: bins (either all, or above cutoff only)
        
        Returns: pandas Series with labels:
        >=0 for trips in bins. the label is a unique bin id
        =-1 for long trips not in bins
        =-2 for trips that were too short
        """
        # This is a bit tricky wrt indexing, since the indices of the trips in the bin are after filtering,
        # so don't match up 1:1 with the indices in the trip dataframe
        # since we create a new dataframe for the filtered trips, they should match up with the filtered dataframe
        # but the index of the filtered dataframe is a new RangeIndex, so it doesn't work for indexing into the result series
        # so we need to follow a two-step process as below

        all_trip_df = pd.DataFrame(self.all_data)
        if hasattr(self, "filtered_data"):
            filtered_trip_df = pd.DataFrame([e for e in self.filtered_data])
            # print(filtered_trip_df)
        else:
            filtered_trip_df = None

        # logging.debug(f"lengths: {len(all_trip_df)}, {len(filtered_trip_df) if filtered_trip_df is not None else None}")

        # assume that everything is noise to start with
        result_labels = pd.Series([-1] * len(all_trip_df), dtype="int")

        # self.bins contains the bins in the current state (either before or
        # after cutoff). Loop will not run if binning is not complete, so all
        # trips will be noise
        for i, curr_bin in enumerate(self.bins):
            if filtered_trip_df is not None and len(filtered_trip_df) > 0:
                # get the trip ids of matching filtered trips for the current bin
                matching_filtered_trip_ids = filtered_trip_df.loc[curr_bin]._id
                # then, match by tripid to find the corresponding entries in the all_trips dataframe
                matching_all_trip_ids = all_trip_df[all_trip_df._id.isin(
                    matching_filtered_trip_ids)].index
                # then set all those indices to the bin index
                result_labels.loc[matching_all_trip_ids] = i
            else:
                # No filteration, trip indices in the bins are from all trips
                # so we can index directly
                result_labels.loc[curr_bin] = i

        # For now, we also mark the "too short" labels with -2 to help with
        # our understanding. Consider removing this code later. This will override
        # noisy labels
        if filtered_trip_df is not None and len(filtered_trip_df) > 0:
            removed_trips = all_trip_df[~all_trip_df._id.isin(filtered_trip_df.
                                                              _id)]
            logging.debug("Detected removed trips %s" % removed_trips.index)
            result_labels.loc[removed_trips.index] = -2
        return result_labels

    #calculate the cut-off point in the histogram
    #This is motivated by the need to calculate the cut-off point
    #that separates the common trips from the infrequent trips.
    #This works by approximating the point of maximum curvature
    #from the curve formed by the points of the histogram. Since
    #it is a discrete set of points, we calculate the point of maximum
    #distance from the line formed by connecting the height of the
    #tallest bin with that of the shortest bin, as described
    #here: http://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve?lq=1
    #We then remove all bins of lesser height than the one chosen.
    def elbow_distance(self):
        y = [0] * len(self.bins)
        for i in range(len(self.bins)):
            y[i] = len(self.bins[i])
        N = len(y)
        x = list(range(N))
        max = 0
        index = -1
        a = np.array([x[0], y[0]])
        b = np.array([x[-1], y[-1]])
        n = norm(b - a)
        new_y = []
        for i in range(0, N):
            p = np.array([x[i], y[i]])
            dist = old_div(norm(np.cross(p - a, p - b)), n)
            new_y.append(dist)
            if dist > max:
                max = dist
                index = i
        return index

    #check if two trips match
    # unused
    def match(self, a, bin, loc_type):
        """ Args:
                a (int): index of trip in the dataframe
                bin (int list): list of trip indices
                loc_type (str): either 'start' or 'end'
        """
        for b in bin:
            if not self.distance_helper(a, b, loc_type):
                return False
        return True

    #evaluate the bins as if they were a clustering on the data
    def evaluate_bins(self):
        self.labels = []
        for bin in self.bins:
            for b in bin:
                self.labels.append(self.bins.index(bin))
        if not self.data or not self.bins:
            return
        if len(self.labels) < 2:
            logging.debug('Everything is in one bin.')
            return
        labels = np.array(self.labels)
        points = []
        for bin in self.bins:
            for b in bin:
                tb = self.data[b]
                start_lon = tb.data.start_loc["coordinates"][0]
                start_lat = tb.data.start_loc["coordinates"][1]
                end_lon = tb.data.end_loc["coordinates"][0]
                end_lat = tb.data.end_loc["coordinates"][1]
                path = [start_lat, start_lon, end_lat, end_lon]
                points.append(path)
        logging.debug("number of labels are %d, number of points are = %d" %
                      (len(labels), len(points)))
        a = metrics.silhouette_score(np.array(points), labels)
        logging.debug('number of bins is %d' % len(self.bins))
        logging.debug('silhouette score is %d' % a)
        return a

    #calculate the distance between two trips
    def distance_helper(self, a, b, loc_type):
        """ modified from the original class to consider only start or end points, not both 
        
            Args:
                a (int): index of trip in the dataframe
                b (int): index of another trip in the dataframe
                loc_type (str): either 'start' or 'end'
        """
        tripa = self.data_df.iloc[a]
        tripb = self.data_df.iloc[b]

        pta_lat = tripa[[loc_type + '_lat']]
        pta_lon = tripa[[loc_type + '_lon']]
        ptb_lat = tripb[[loc_type + '_lat']]
        ptb_lon = tripb[[loc_type + '_lon']]

        if loc_type == 'start':
            return within_radius(pta_lat, pta_lon, ptb_lat, ptb_lon,
                                 self.radius_start)
        elif loc_type == 'end':
            return within_radius(pta_lat, pta_lon, ptb_lat, ptb_lon,
                                 self.radius_end)

    def bin_helper(self, loc_type=None):
        """ generates two sets of bins, one for start points and one for end points.
        """
        for a in range(len(self.data_df)):
            # print(a)
            # a is the index of the trip in the dataframe
            added_start = False
            added_end = False
            # right now we are basically only using self.start_bins and self.
            # end_bins to count the number of bins we have. eventually we
            # should remove it and replace it with an int variable
            if loc_type == None or loc_type == 'start':
                for i in range(len(self.start_bins)):
                    # i is the index of the bin
                    bin = self.start_bins[i]
                    try:
                        if self.match(a, bin, 'start'):
                            self.data_df.loc[a, 'start_bin'] = i
                            # keeping list of bins for historical purposes for now
                            bin.append(a)
                            added_start = True
                            break
                    except Exception as e:
                        raise e
                        print(repr(e))
                        print(
                            f"Exception occurred. Start of trip {a} does not fit in existing bins, creating new bin. Theoretically I don't see why we should ever reach this though?"
                        )
                        added_start = False
                if not added_start:
                    self.data_df.loc[a, 'start_bin'] = len(self.start_bins)
                    self.start_bins.append([a])

            if loc_type == None or loc_type == 'end':

                for i in range(len(self.end_bins)):
                    bin = self.end_bins[i]
                    try:
                        if self.match(a, bin, 'end'):
                            self.data_df.loc[a, 'end_bin'] = i
                            # keeping list of bins for historical purposes for now
                            bin.append(a)
                            added_end = True
                            break
                    except Exception as e:
                        raise e
                        print(repr(e))
                        print(
                            f"Exception occurred. End of trip {a} does not fit in existing bins, creating new bin. Theoretically I don't see why we should ever reach this though?"
                        )
                        added_end = False
                if not added_end:
                    self.data_df.loc[a, 'end_bin'] = len(self.end_bins)
                    self.end_bins.append([a])

        self.start_bins.sort(key=lambda bin: len(bin), reverse=True)
        self.end_bins.sort(key=lambda bin: len(bin), reverse=True)

    def group_trips(self):
        """ adds new column called 'trip_bin' to data_df """
        assert 'start_bin' in self.data_df.columns
        assert 'end_bin' in self.data_df.columns

        all_combos = self.data_df.groupby(["start_bin", "end_bin"])
        all_combos_dict = dict(all_combos.groups)
        all_combos_series = pd.Series(list(all_combos_dict.keys()))

        for group_idx in range(len(all_combos_series)):
            group_tuple = all_combos_series[group_idx]
            idxlist = all_combos_dict[group_tuple]
            self.data_df.loc[idxlist, "trip_bin"] = group_idx
