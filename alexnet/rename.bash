#!/bin/bash 

# Rename all *.txt to *.text
cd images/

for f in *.jpeg; do 
    mv -- "$f" "${f%.jpeg}.jpg"
done


do_rename() {
    (
    export prevdir="$(pwd)"
    cd "$1"
    for file in *
    do
        if [ -d "$file" ]
        then
            echo "$file is a directory, renaming recursively"
            do_rename "$file"
        elif [ -f "$file" ]
        then
            dirpath="$prevdir/$1"
            oldname="$dirpath/$file"
            name_hash=$(echo "$oldname" | md5sum)
            newname="$dirpath/${name_hash:0:32}.jpg"
            echo "mv $oldname $newname"
            mv $oldname $newname
        fi
    done
    cd "$prevdir"
    )
}

do_rename "$1"


a=1
for i in *.jpg; do
  new=$(printf "image_%d.jpg" "$a") #04 pad to length of 4
  mv -i -- "$i" "$new"
  let a=a+1
done
