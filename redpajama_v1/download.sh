# wget 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'
while read line; do
    dload_loc=${line#https://data.together.xyz/redpajama-data-1T/v1.0.0/}
    mkdir -p $(dirname $dload_loc)
    wget "$line" -O "$dload_loc"
done < urls.txt
