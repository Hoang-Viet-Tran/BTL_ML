# IMDb Movie Analysis Project

This repository contains a preprocessing and feature engineering pipeline for IMDb movie datasets. 

## Data Source

The raw data can be downloaded from the official IMDb datasets page:  
[https://datasets.imdbws.com/](https://datasets.imdbws.com/)

### IMDb Non-Commercial Datasets

Subsets of IMDb data are available for access to customers for **personal and non-commercial use**.  
You can hold local copies of this data, and it is subject to IMDb terms and conditions.  
Please refer to the Non-Commercial Licensing and copyright/license and verify compliance.

### Notice

As of March 18, 2024, the datasets on this page are backed by a new data source.  
There has been no change in location or schema, but if you encounter issues with the datasets following the March 18th update, please contact `imdb-data-interest@imdb.com`.

### Data Location

The dataset files can be accessed and downloaded from [https://datasets.imdbws.com/](https://datasets.imdbws.com/). The data is refreshed daily.

### IMDb Dataset Details

Each dataset is contained in a gzipped, tab-separated-values (TSV) formatted file in UTF-8 character set.  
The first line in each file contains headers that describe what is in each column.  
A `\N` is used to denote missing or null fields.

#### Available Datasets

- **title.akas.tsv.gz**  
  - `titleId` (string) - a tconst, an alphanumeric unique identifier of the title  
  - `ordering` (integer) – a number to uniquely identify rows for a given titleId  
  - `title` (string) – localized title  
  - `region` (string) – region for this version of the title  
  - `language` (string) – language of the title  
  - `types` (array) – enumerated set of attributes for this alternative title  
  - `attributes` (array) – additional terms to describe this alternative title  
  - `isOriginalTitle` (boolean) – 0: not original; 1: original  

- **title.basics.tsv.gz**  
  - `tconst` (string) - alphanumeric unique identifier of the title  
  - `titleType` (string) – type/format of the title (e.g., movie, short, tvseries)  
  - `primaryTitle` (string) – popular title used for promotion  
  - `originalTitle` (string) – original title  
  - `isAdult` (boolean) – 0: non-adult, 1: adult  
  - `startYear` (YYYY) – release year  
  - `endYear` (YYYY) – end year for TV series, '\N' otherwise  
  - `runtimeMinutes` – primary runtime in minutes  
  - `genres` (string array) – up to three genres  

- **title.crew.tsv.gz**  
  - `tconst` (string) - alphanumeric unique identifier of the title  
  - `directors` (array of nconsts) – director(s)  
  - `writers` (array of nconsts) – writer(s)  

- **title.episode.tsv.gz**  
  - `tconst` (string) – unique episode ID  
  - `parentTconst` (string) – parent TV series ID  
  - `seasonNumber` (integer) – season number  
  - `episodeNumber` (integer) – episode number  

- **title.principals.tsv.gz**  
  - `tconst` (string) – unique title ID  
  - `ordering` (integer) – row order for given titleId  
  - `nconst` (string) – unique person ID  
  - `category` (string) – job category  
  - `job` (string) – specific job title  
  - `characters` (string) – character played  

- **title.ratings.tsv.gz**  
  - `tconst` (string) – unique title ID  
  - `averageRating` – weighted average rating  
  - `numVotes` – number of votes  

- **name.basics.tsv.gz**  
  - `nconst` (string) – unique person ID  
  - `primaryName` (string) – name most often credited  
  - `birthYear` (YYYY) – birth year  
  - `deathYear` (YYYY) – death year if applicable, else '\N'  
  - `primaryProfession` (array of strings) – top-3 professions  
  - `knownForTitles` (array of tconsts) – titles the person is known for
