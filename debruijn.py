#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
from pathlib import Path
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw,
    spring_layout,
)
import matplotlib
from operator import itemgetter
import random

random.seed(9001)
from random import randint
import statistics
import textwrap
import matplotlib.pyplot as plt
from typing import Iterator, Dict, List

matplotlib.use("Agg")

__author__ = "Aggar Emma"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Aggar Emma"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Aggar Emma"
__email__ = "emmaag2003.ea@gmail.com"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, 'rt') as debruijn :
        for line in debruijn :
            yield next(debruijn).replace('\n',"")
            next(debruijn)
            next(debruijn)


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(0, len(read)-kmer_size+1):
        yield read[i:i+kmer_size]


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    dico = {}
    for seq in read_fastq(fastq_file):
        for kmer in cut_kmer(seq, kmer_size):
            if(kmer in dico):
                dico[kmer]+=1
            else:
                dico[kmer]=1
    return dico


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    G = DiGraph()
    for seq in kmer_dict :
        G.add_edge(seq[:-1], seq[1:], weight=kmer_dict[seq])
    return G


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list : 
        if delete_entry_node : 
            graph.remove_node(path[0])
        if delete_sink_node: 
            graph.remove_node(path[-1])
        for node in path[1:-1]:
            graph.remove_node(node)
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    indice = 0
    for i in range (1, len(path_list)):
        if weight_avg_list[i] > weight_avg_list[indice] : 
            indice = i
        elif weight_avg_list[i] == weight_avg_list[indice] : 
            if path_length[i] > path_length[indice]:
                indice = i
            elif path_length[i] == path_length[indice]:
                indice = random.choice([indice,i])

    remove = [path for i, path in enumerate(path_list) if i!=indice]
    graph = remove_paths(graph, remove, delete_entry_node=delete_entry_node, delete_sink_node=delete_sink_node)
    return graph

def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    liste = list(all_simple_paths(graph, ancestor_node, descendant_node))
    if len(liste)==0 or len(liste)==1:
        return graph
    longueur = [len(path) for path in liste]
    poids = [
        path_average_weight(graph, path)
        for path in liste
    ]
    graph = select_best_path(graph=graph, path_list=liste, path_length=longueur, weight_avg_list=poids,delete_entry_node=False, delete_sink_node=False)
    return graph

def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    bubble = False
    for node in list(graph.nodes):
        liste_p = list(graph.predecessors(node))
        if len(liste_p) >1:
            for i in range(len(liste_p)):
                for j in range(i+1, len(liste_p)):
                    ancestor_node = lowest_common_ancestor(graph, liste_p[i], liste_p[j])
                    if ancestor_node is not None : 
                        bubble = True 
                        graph = solve_bubble(graph, ancestor_node, node)
                        break
                if bubble:
                    break
        if bubble:
            break
    if bubble:
        graph = simplify_bubbles(graph)
    return graph


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for start in starting_nodes :
        if start not in graph:
            continue
        liste_s = list(graph.successors(start))
        for s in liste_s : 
            liste_p = list(graph.predecessors(s))
            if len(liste_p) > 1:
                paths = [ [p,s] for p in liste_p]
                longueur = [len(path) for path in paths ]
                poids = [path_average_weight(graph, path) for path in paths]
                graph = select_best_path(graph, path_list=paths, path_length=longueur, weight_avg_list=poids, delete_entry_node=True, delete_sink_node=False)
        return graph



def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for end in ending_nodes : 
        if end not in graph :
            continue
        liste_p = list(graph.predecessors(end))
        for p in liste_p:
            liste_s = list(graph.successors(p))
            if len(liste_s) > 1:
                paths = [[p,s] for s in liste_s]
                longueur = [len(path) for path in paths]
                poids = [path_average_weight(graph, path) for path in paths]
                graph = select_best_path(graph, path_list=paths, path_length=longueur, weight_avg_list=poids, delete_entry_node=False, delete_sink_node=True)
        return graph



def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    liste = []
    for node in graph:
        if len(list(graph.predecessors(node)))==0:
            liste.append(node)
    return liste


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    liste = []
    for node in graph:
        if len(list(graph.successors(node)))==0:
            liste.append(node)
    return liste


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    c = []
    for s in starting_nodes:
        for e in ending_nodes:
            if has_path(graph, s, e) :
                for path in all_simple_paths(graph, s, e):
                    c0 =  path[0]
                    for node in path[1:]:
                        c0 += node[-1]
                    c.append((c0, len(c0)))
    return c


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, "w", newline='\n') as file :
        for i, (c0, length) in enumerate(contigs_list):
            file.write(f">contig_{i} len={length}\n")
            #file.write(f"{textwrap.fill(c0, width=80)}\n")
            wrapped_c0 = textwrap.fill(c0, width=80)
            file.write(f"{wrapped_c0}\n")
        #file.close()


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Fonctions de dessin du graphe
    # A decommenter si vous souhaitez visualiser un petit
    # graphe
    # Plot the graph
    # if args.graphimg_file:
    #     draw_graph(graph, args.graphimg_file)
    
    print("Construction du dictionnaire de k-mers")
    kmer_dict = build_kmer_dict(args.fastq_file, args.kmer_size)
    print("Construction du graphe de De Bruijn")
    graph = build_graph(kmer_dict)
    print("Simplification des bulles")
    graph = simplify_bubbles(graph)
    print("Résoluion des pointes d'entrée")
    start = get_starting_nodes(graph)
    graph = solve_entry_tips(graph, start)
    print("Résolution des pointes de sortie")
    end = get_sink_nodes(graph)
    graph = solve_out_tips(graph, end)
    print("Extraction des contigs")
    start = get_starting_nodes(graph)
    end = get_sink_nodes(graph)
    c = get_contigs(graph, start, end)
    print(f"Ecriture des contigs dans le fichier {args.output_file}")
    save_contigs(c, args.output_file)
    if args.graphimg_file:
        print(f"Enregistrement du graphe sous forme d'image : {args.graphimg_file}")
        draw_graph(graph, args.graphimg_file)
    print("Assemblage terminé")
    

if __name__ == "__main__":  # pragma: no cover
    main()
