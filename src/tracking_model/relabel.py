import os
import re
import numpy as np
import networkx as nx
import tifffile as tiff
from .GeometricInterpolation import InterpolateTubes, DrawContourForLabel
from .io_utils_el import write_image
from .Label_read import *

# Lineage tree

##########################################################
# Read input file...
# Input:
# csv_file - file with lineage tree information
# Output:
# lines_labels - labels depicting nodes
# lines_relationships - labels depicting node connections
##########################################################
def read_tree_file(csv_file):
    with open(csv_file, 'r') as file:
        lines = file.readlines()

    lines_labels = []
    lines_relationships = []

    i = 0
    while i < len(lines):
        # Read each line
        line_curr = lines[i].strip().split(',')
        cline_part1, cline_part2 = map(str.strip, line_curr)

        # Read the next line
        if i < len(lines) - 1:
            line_next = lines[i + 1].strip().split(',')
            nline_part1, nline_part2 = map(str.strip, line_next)

        # Identify divisions
        # A label is divided in two new ones
        if i < len(lines) - 1 and cline_part1 == nline_part1:
            lines_relationships.extend([line_curr, line_next])
            i += 2
        # Labels depicting the same node
        else:
            lines_labels.append(line_curr)
            i += 1
    return lines_labels, lines_relationships

##############################################################
# Identify same labels...
# Input:
# lines_labels - labels depicting nodes
# Output:
# connected_labels - groups of labels depicting the same node
##############################################################
def store_connected_labels(lines_labels):
    
    # Linked list node
    class Node:
        def __init__(self, data):
            self.data = data
            self.next = None
        
    connected_labels = []
    node_dict = {}

    for line in lines_labels:
        # Get labels
        node_a = line[0].strip()
        node_b = line[1].strip()

        # First label
        if node_a not in node_dict:
            # New label
            new_node = Node(node_a)
            node_dict[node_a] = new_node
        else:
            new_node = node_dict[node_a]

        # Second label
        if node_b not in node_dict:
            # New label
            next_node = Node(node_b)
            node_dict[node_b] = next_node
        else:
            next_node = node_dict[node_b]

        # Point to the 2nd label
        if new_node.next is None:
            new_node.next = [next_node]
        else:
            new_node.next.append(next_node)

    visited = set()
    for head in node_dict.values():
        # Check if done
        if head not in visited:
            current_node = head
            conn_list = []
            while current_node is not None:
                visited.add(current_node)
                conn_list.append(current_node.data)
                # All labels in this list
                current_node = current_node.next[0] if current_node.next else None
            # Add this group of labels
            connected_labels.append(conn_list)

    return connected_labels

#########################################################################################
# Process labels relationships and IDs...
# Input:
# lines_relationships - labels depicting node connections
# connected_labels - groups of labels depicting the same node 
# Output:
# connected_labels_mapping - groups of labels depicting the same node with disctinct ids
# graph - graph depicting the lineage tree (with the label groups as nodes)
#########################################################################################
def process_labels(lines_relationships, connected_labels):
    
    # Create graph
    graph = nx.DiGraph()
    for i in range(len(connected_labels)):
        # Add group ids as nodes
        graph.add_node(f"{i}") 
    
    for line in lines_relationships:
        # Get labels
        node_a = line[0].strip()
        node_b = line[1].strip()

        component_a = None
        component_b = None

        for i, label_component in enumerate(connected_labels):
            # The node id will be the group's id
            if node_a in label_component:
                component_a = i
            if node_b in label_component:
                component_b = i

        # If the label does not exist in any group, add it to a new group
        if component_a is None:
            connected_labels.append([node_a])
            component_a = len(connected_labels) - 1
            # Add new node (group id) to the graph
            graph.add_node(f"{component_a}") 

        if component_b is None:
            connected_labels.append([node_b])
            component_b = len(connected_labels) - 1
            # Add new node (group id) to the graph
            graph.add_node(f"{component_b}") 

        if component_a != component_b:
            # Connect the two nodes
            graph.add_edge(f"{component_a}", f"{component_b}") 
    
    # Assign ids to the groups of labels
    connected_labels_mapping = {}
    for i, label_component in enumerate(connected_labels):
        connected_labels_mapping[i] = label_component
    
    return graph, connected_labels_mapping

##################################################################################################
# Function that given a pair of images, relabels the 2nd image according to the cell lineage tree
# Input:
# image_timestamp_i - 3D image timestamp i
# image_timestamp_ii - 3D image timestamp i+1
# output_folder - location to save the output relabelled image
# graph - graph depicting the lineage tree (with the label groups as nodes)
# connected_labels_mapping - groups of labels depicting the same node with disctinct ids
# Output:
# Relabelled image i+1 tiff file
##################################################################################################
def relabel_2nd_image(input_image_i, input_image_ii, output_folder, graph, connected_labels_mapping):

    # Get image information
    pattern = r'(\d{3})\b'
    time_index_i = re.findall(pattern, input_image_i)[-1]
    time_index_ii = re.findall(pattern, input_image_ii)[-1]
    print("Timestamps: ", time_index_i, "-", time_index_ii)

    # Get labels of 2nd image
    image_labels = read_image(input_image_ii)
    labels = np.unique(image_labels)
    print("Number of labels: ", len(labels))

    # Change the labels of the 2nd image
    new_image_labels = np.zeros(image_labels.shape,dtype=np.uint16)

    for ilabel in labels:
        if ilabel != 0:
            #print("---", ilabel)
            ind = np.where(image_labels == ilabel)

            flabel = "{:03d}".format(ilabel)
            search_label = time_index_ii + "_" + flabel
            search_label_prev = time_index_i + "_"

            ####################
            new_ilabel = []
            
            for node_key, node_labels in connected_labels_mapping.items(): 
                if search_label in node_labels:
                    get_label_i = [i for i in node_labels if search_label_prev in i]
                    
                    # get label with time_index_i from current node - same cell
                    if(get_label_i != []):
                        #print("Same cell...")
                        new_ilabel = get_label_i

                    # get label with time_index_i from predecessor node - mother cell
                    else:
                        #print("Looking for mother cell...")
                        for p_nodekey in graph.predecessors(str(node_key)):
                            p_node_labels = connected_labels_mapping[int(p_nodekey)]
                            get_label_i = [i for i in p_node_labels if search_label_prev in i]
                            if(get_label_i != []):
                                new_ilabel = get_label_i
                                
                            else:
                                print("ERROR: No label found!")

            mod_new_ilabel = np.uint16(int((new_ilabel[0]).split('_')[1]))          
            new_image_labels[ind] = mod_new_ilabel
            ####################

    # New label image
    return new_image_labels, time_index_ii

################## Run ##################
def run_relabelling(csv_file0: os.PathLike, input_image1: os.PathLike, input_image2: os.PathLike, outfolder: os.PathLike):
    
    print("Relabel...")
    
    # Read cell lineage tree file
    lines_labels0, lines_relationships0 = read_tree_file(csv_file0)

    # Labels
    connected_labels0 = store_connected_labels(lines_labels0)
    
    # Graph
    graph0, connected_labels_mapping0 = process_labels(lines_relationships0, connected_labels0)
    
    # Relabel images
    new_image_labels0, time_index_ii0 = relabel_2nd_image(input_image1, input_image2, outfolder, graph0, connected_labels_mapping0)

    # Save new label image
    print("Saving image...")
    new_image_labels_cont = np.ascontiguousarray(new_image_labels0)
    write_image(new_image_labels_cont, os.path.join(outfolder, "relabel_" + str(time_index_ii0)), 'TIF') #'KLB'
    print("Image saved: ", time_index_ii0)
    
    return outfolder


