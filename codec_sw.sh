# Change encode
echo "iconv: ascii to utf-8..."
time iconv -f GBK -c -t UTF-8 < SogouR.txt > wds_grp_utf-8.txt
