# <table style="width:100%; background-color: #D9EDF7">
#   <tr>
#     <td style="border: 1px solid #CFCFCF">
#       <b>Renewable power plants: Main Notebook</b>
#       <ul>
#         <li>Main Notebook</li>
#         <li><a href="download_and_process.ipynb">Download and process Notebook</a></li>
#         <li><a href="validation_and_output.ipynb">Validation and output Notebook</a></li>
#       </ul>
#       <br>This Notebook is part of the <a href="http://data.open-power-system-data.org/renewable_power_plants"> Renewable power plants Data Package</a> of <a href="http://open-power-system-data.org">Open Power System Data</a>.
#     </td>
#   </tr>
# </table>
# 

# # 1. About Open Power System Data 
# This Notebook is part of the project [Open Power System Data](http://open-power-system-data.org). Open Power System Data develops a platform for free and open data for electricity system modeling. We collect, check, process, document, and provide data that are publicly available but currently inconvenient to use. 
# More info on Open Power System Data:
# - [Information on the project on our website](http://open-power-system-data.org)
# - [Data and metadata on our data platform](http://data.open-power-system-data.org)
# - [Data processing scripts on our GitHub page](https://github.com/Open-Power-System-Data)
# 
# # 2. About Jupyter Notebooks and GitHub
# This file is a [Jupyter Notebook](http://jupyter.org/). A Jupyter Notebook is a file that combines executable programming code with visualizations and comments in markdown format, allowing for an intuitive documentation of the code. We use Jupyter Notebooks for combined coding and documentation. We use Python 3 as programming language. All Notebooks are stored on [GitHub](https://github.com/), a platform for software development, and are publicly available. More information on our IT-concept can be found [here](http://open-power-system-data.org/it). See also our [step-by-step manual](http://open-power-system-data.org/step-by-step) how to use the dataplatform.
#  
# # 3. About this Data Package
# We provide data in different chunks, or [Data Packages](http://frictionlessdata.io/data-packages/). The one you are looking at right now, [Renewable power plants](http://data.open-power-system-data.org/renewable_power_plants/), contains
# 
# * lists of renewable energy power plants of Germany, Denmark, France and Poland
# * daily time series of cumulated installed capacity per energy source type for Germany.
# 
# Due to differing data availability, the power plant lists are of variable accurancy and partly provide different power plant parameters. Therefore the lists are provided as separate CSV files per country and as separate sheets in the Excel file.
# 
# * Germany: More than 1.7 million power plant entries, all renewable energy plants supported by the German Renewable Energy Law (EEG)
# * Denmark: Wind and phovoltaic power plants with a high level of detail
# * France: Summed capacity and number of installations per energy source per municipality (Commune)
# * Poland: Summed capacity and number of installations per energy source per municipality (Powiat)
#  
# # 4. Data sources
# 
# This Data Package uses the following main data sources.
# ## 4.1 Germany - DE
# - [Netztransparenz.de](https://www.netztransparenz.de/de/Anlagenstammdaten.htm) - Information platform from the German TSOs
# 
# >In Germany historically all data has been published mandatorily by the four TSOs (50Hertz, Amprion, Tennet, TransnetBW). This obligation expired in August 2014, nonetheless the TSO reported until the end of 2014 and issued another update in August 2016 for plants commissioned until end of 2015 (which is used in this script).
# 
# - [BNetzA](http://www.bundesnetzagentur.de/) - The German Federal Network Agency for Electricity, Gas, Telecommunications, Posts and Railway
# 
# >Since August 2014 the BNetzA is responsible to publish the renewable power plants register. The legal framework for the register is  specified in the EEG 2014 [(German)](http://www.gesetze-im-internet.de/eeg_2014/) [(English)](http://www.res-legal.eu/search-by-country/germany/single/s/res-e/t/promotion/aid/feed-in-tariff-eeg-feed-in-tariff/lastp/135/). All power plants are listed in a new format: two separate MS-Excel and CSV files for roof-mounted PV power plants ["PV-Datenmeldungen"](http://www.bundesnetzagentur.de/cln_1422/DE/Sachgebiete/ElektrizitaetundGas/Unternehmen_Institutionen/ErneuerbareEnergien/Photovoltaik/DatenMeldgn_EEG-VergSaetze/DatenMeldgn_EEG-VergSaetze_node.html) and all other renewable power plants [" Anlagenregister"](http://www.bundesnetzagentur.de/cln_1412/DE/Sachgebiete/ElektrizitaetundGas/Unternehmen_Institutionen/ErneuerbareEnergien/Anlagenregister/Anlagenregister_Veroeffentlichung/Anlagenregister_Veroeffentlichungen_node.html).
# 
# ## 4.2 Denmark - DK
# 
# - [ens.dk](http://www.ens.dk/) - Energy Agency Denmark 
#     
# >The Danish Energy Agency publishes a national master data register for wind turbines which was created in collaboration with the transmission system operators. The publication is monthly as an Excel file. The data set includes all electricity-generating wind turbines with information about technical data, location data and production data.
# 
# 
# - [Energinet.dk](http://www.energinet.dk/EN/Sider/default.aspx) - Transmission system Operator in Denmark
# 
# >The photovoltaic statistic, published from Energinet, includes information about location, year of implementing, installed capacity and number of systems. There is an additional overview of the number of plants and installed capacity per  postcode. The publication an  Excel file and dates of publication are not entirely clear.
# 
# ## 4.3 France - FR
# - [Ministery of the Environment,Energy and the Sea France](http://www.developpement-durable.gouv.fr/)
# 
# > The data is annual published on the france [website for statistics](http://www.statistiques.developpement-durable.gouv.fr/energie-climat/r/energies-renouvelables.html?tx_ttnews[tt_news]=20647) as an Excel file. The Excel chart includes number and installed capacity of the different renewable source for every municipality in France. It is limited to the plants which are covered by article 10 of february 2000 by an agreement to a purchase commitment.
# 
# ## 4.4 Poland - PL
# - [Urzad Regulacji Energetyki (URE)](http://www.ure.gov.pl/uremapoze/mapa.html) - Energy Regulatory Office of Poland
# 
# > Number of installations and installed capacity per energy source of renewable energy, summed per powiat (districts) is illustrated on the page and can be downloaded as rtf-file
# 
# A complete list of data sources is provided on the [Data Package information website](http://data.open-power-system-data.org/renewable_power_plants/). They are also contained in the JSON file that contains all metadata.
# 
# ## 4.5 Switzerland - CH
# - [Swiss Federal Office of Energy](http://www.bfe.admin.ch/themen/00612/02073/index.html?dossier_id=02166&lang=de)
# 
# >Data of all renewable power plants receiving "Kostendeckende Einspeisevergütung" (KEV) which is the Swiss feed in tarif for renewable power plants. 
# Geodata is based on municipality codes.
#  
# # 5. Naming Conventions
# ## 5.1 Column translation list
# This list provides all internal translations of column original names to the OPSD standard names in order to achieve common data structure for data of all implemented countries as well as the other data packages. 
# 

import pandas as pd
pd.read_csv('input/column_translation_list.csv')


# ## 5.2 Value translation list
# 
# This list provides all internal translations of original value names to the OPSD standard names in order to achieve common data structure for data of all implemented countries as well as the other data packages. 
# 

import pandas as pd
pd.read_csv('input/value_translation_list.csv')


# ## 5.3 Validation marker
# 
# Validation markers are used in comments column in order to mark units for which we identified one of the following issues:
# 

import pandas as pd
pd.read_csv('input/validation_marker.csv')


# ## 5.4 Energy source structure
# ![OPSD-Tree](http://open-power-system-data.org/2016-10-25-opsd_tree.svg)
#  
# # 6. License
# This Jupyter Notebook as well as all other documents in this repository is published under the [MIT License](LICENSE.md).

