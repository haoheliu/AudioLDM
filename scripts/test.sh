# Generation
audioldm --file_path trumpet.wav
audioldm --text "A hammer is hitting a wooden surface"
audioldm

# False use cases
audioldm --text "A hammer is hitting a wooden surface" --file_path trumpet.wav # Same as audioldm --file_path trumpet.wav


# Transfer
audioldm --mode "transfer" --file_path trumpet.wav -t "Children Singing" 



