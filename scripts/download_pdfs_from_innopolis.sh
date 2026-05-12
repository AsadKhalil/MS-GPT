#!/bin/bash
#124578963
# Define the source directory and destination directory
source_dir="asad@172.21.136.243:/tear/dasulimov/home_folder/mass_spec_papers_multi/"
destination_dir="/home/tk-lpt-0806/Desktop/pdf_to_process/input/"

# List of file names
files=(
"05a12eb3b6ce838fd91e1a38aea7e743cec5c941.pdf"
"0b5c8d99b0dc16f850f3091db5eb7fe802481889.pdf"
"10fb849f99b350910fc37aa09913ad703031c17d.pdf"
"6e42beb1595ef78ee1e3733755aa218eccdced52.pdf"
"05a1646dacc7c3e34781eee08516b09d1980ce8f.pdf"
"0b5cc3452143a1f121b28847ed26b944351d28ca.pdf"
"10fc965726c2acd5adfa68171ec4bfccfd0f1bcc.pdf"
"6e43af6bab1b82479b0a56d811f6299e24238dba.pdf"
# Add other files here as needed
)

# Loop through the files and download them
for file in "${files[@]}"; do
    scp "${source_dir}${file}" "${destination_dir}"
done


# put folder to server
# scp -r /home/tk-lpt-0806/Desktop/rag asad@172.21.136.243:/home/asad