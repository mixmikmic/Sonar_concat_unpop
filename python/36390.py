# # Long-Term Unemployment
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns


unem = pd.read_table("econ-data/une_rt_a.tsv")
ltunem = pd.read_table("econ-data/une_ltu_a.tsv")


unem


ltunem


# # WUDC Outround Opponent Finder
# 
# Function of interest is `outround_opponents_full(team_break_position)`.
# 
# Sample data is from WUDC 2017.
# 

import pandas as pd


# Read in list of breaking teams (in order of breaking)

name = 'T'
breaking_teams = pd.read_csv("wudc-2017-open-break.csv", names=name)
breaking_teams_list = [
    breaking_teams.loc[i][name]
    for i in range(len(breaking_teams))
]


# Parameters

total_teams = 48


def calculate_outround_mods(team_break_position):
    """Calculate mods of teams a given team could face per outround
    given said team's breaking position. See 'Reasoning' in the Appendix.
    """
    break_position_mod_16 = team_break_position % 16
    pdo_mods = sorted([break_position_mod_16, (break_position_mod_16 * -1 + 1) % 16])
    quarter_mods = sorted([pdo_mods[0] % 8, pdo_mods[1] % 8])
    semi_mods = sorted([pdo_mods[0] % 4, pdo_mods[1] % 4])
    return pdo_mods, quarter_mods, semi_mods


def outround_team_indices(round_mod_list, mod):
    """Return opponents' zero-indexed break positions given a round
    mod list and a round mod.
    """
    return sorted([
        round_mod_list[j] + mod * i
        for i in range(total_teams // mod)
        for j in range(2)
    ])

def octo_team_indices(octo_mod_list):
    """Return octofinal opponents' zero-indexed break positions."""
    return outround_team_indices(octo_mod_list, 16)

def quarter_team_indices(quarter_mod_list):
    """Return quarterfinal opponents' zero-indexed break positions."""
    return outround_team_indices(quarter_mod_list, 8)

def semi_team_indices(semi_mod_list):
    """Return semifinal opponents' zero-indexed break positions."""
    return outround_team_indices(semi_mod_list, 4)


def pdo_from_octo(octo_indices):
    """Return PDO opponents' zero-indexed break positions from octo
    opponents' zero-indexed break positions."""
    if octo_indices[0] == 0:
        return octo_indices[0] + octo_indices[3:]
    else:
        return octo_indices[2:]


def lookup_team_names(team_indices):
    """Return team names given their zero-indexed break positions."""
    return [
        breaking_teams_list[i-1]
        for i in team_indices
    ]


def outround_opponents_full(team_break_position):
    """Print opponents team could face in all outrounds."""
    print("Team: ", breaking_teams_list[team_break_position-1])
    print("Break position: ", team_break_position)
    octo_mods, quarter_mods, semi_mods = calculate_outround_mods(team_break_position)
    octo_indices = octo_team_indices(octo_mods)
    if team_break_position > 16:
        print("\nPDO opponents: ", lookup_team_names(pdo_from_octo(octo_indices)))
    print("\nOcto opponents: ", lookup_team_names(octo_indices))
    print("\nQuarter opponents: ", lookup_team_names(quarter_team_indices(quarter_mods)))
    print("\nSemi opponents: ", lookup_team_names(semi_team_indices(semi_mods)))
    print("\nFinal teams: ", breaking_teams_list)


# Function to return opponents per round

outround_opponents_full(1)


# ## Appendix
# 
# **Reasoning**:
# 
# Octos: 
# * 1, 0 mod 16 
# * (or 2, -1; 3, -2;...;8, -7)
# 
# Quarters: 
# * 1, 0 mod 8 (8, -7)
# * 2, -1 mod 8 (7, -6)
# ...
# 
# Semis:
# * 1, 0 mod 4
# * 2, 3 mod 4
# 

