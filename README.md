# Proxy Card Tools

This project is intended to provide useful tools for improving and formatting various TCG and LCG proxy cards for printing.  It is currently under development so stay tuned for improvements.

*Note: I am currently developing this project to assist with scan improvement and proxy card printing of the Lord of the Rings LCG card game but its underlying methods can likely be applied to more games.*

## Instructions

### Setup

*The following setup information was developed on Windows, alternative setup may be required on Mac and Linux.*

1. Download ModernSpanimationV1.pth from [here](https://openmodeldb.info/models/2x-ModernSpanimationV1) and save it in `external_data/models/`.  It should show up as `external_data/models/2x_ModernSpanimationV1.pth`.
2. Download big-lama.safetensors from [here](https://huggingface.co/michaelgold/big-lama) and save it in `external_data/models/`.  It should show up as `external_data/models/big-lama.safetensors`.
3. Download all Lord of the Rings LCG card scans from [here](https://drive.google.com/drive/folders/1jEy_yvRaPXGylPfilxhweQjffo9AZQsQ).  It should download as three separate zip files.  Extract the files then save them in `/external_data/lotr-lcg-scans/`.
4. Download the Revised Core Set Print and Play PDF from [here](https://images-cdn.fantasyflightgames.com/filer_public/a0/cb/a0cb97ea-a280-4ffa-a9b2-5c72079534dc/core_set_campaign_cards.pdf) and save it in `external_data/print-and-play`.  It should show up as `external_data/print-and-play/core_set_campaign_cards.pdf`.
5. Clone the Lord of the Rings LCG visual assets repository from [here](https://github.com/KevBelisle/lotr-lcg-assets) into the `/external_data/` folder.  It should show up as `/external_data/lotr-lcg-assets/`. *Note: This is not currently used for anything but may be incorporated later.*

### How to Run
See `example/example.py`.  Better instructions are coming later.

## Special Thanks

In no particular order:

- Reddit /u/Apprehensive-Bread37 (GitHub bigtreeproductions) for providing advice and code from their Lord of the Rings LCG AI upscaling project. [GitHub link.](https://github.com/bigtreeproductions/lotr-lcg-scan-clean-up) [Google Drive link.](https://drive.google.com/drive/folders/1d_jQs4Hno0K8e5W1pb8mif5VNFuVzCse)
- Reddit /u/Klutz00 (GitHub KevBelisle) for a maintaining the amazing https://lotr.cardgame.tools website as well as  collection of Lord of the Rings LCG visual assets. [Link here.](https://github.com/KevBelisle/lotr-lcg-assets)
- Reddit /u/axmcxx (GitHub axmcxx) for maintaining the amazing https://proxynexus.net website and providing Lord of the Rings LCG file renaming scripts. [Link here.](https://github.com/axmccx/proxynexus-rs/tree/master/utils/image_file_renamers/lotrlcg)
- Reddit /u/LepcisMagna for a near complete set of 600 DPI Lord of the Rings LCG card scans. [Link here.](https://drive.google.com/drive/folders/1rRKAU5DcQoYqFafdgKBwMNRnTrtl0c3c)
- Reddit /u/Guczini for scanning the remaining Lord of the Rings LCG nightmare cards to make a complete set of all FFG cards.  (Now accessible only accessible on https://lotr.cardgame.tools.)
- JvZijp for making the original Lord of the Rings LCG enhanced proxies and inspiring this project.  [Link here.](https://drive.google.com/drive/folders/1jEy_yvRaPXGylPfilxhweQjffo9AZQsQ)
- Fantasy Flight Games, and their wonderful game designers and artists for creating and supporting the Lord of the Rings LCG for so many years.
- Most of all, **you**, the player.  Keep playing games and supporting this hobby.

## Special Condemnation

- Scalpers.  To anyone taking advantage of our supply issues and selling Lord of the Rings LCG, and other games, at extremely elevated prices.  **Fuck you.**  You are a detriment to gaming, the hobby, and the world.

## How to Support

- Play more games, discuss games, become part of the communities, be nice to people, spread the joy of card games and board games.  Continue being the wonderful people you already are that make this hobby so enjoyable.
- Contribute to this GitHub repo.
- Donate to /u/Klutz00 or /u/axmcxx for maintaining https://lotr.cardgame.tools and https://proxynexus.net.  Web hosting isn't cheap.
- Donate to a local charity.
