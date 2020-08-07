from ProcessedImageCollection import *
import os.path
import argparse


class NodeCollections:

    CLASS_TAG = "[NodeCollections]"

    def __init__(self, root_folder: str, load_cwd_too=True):
        if not os.path.exists(root_folder):
            raise RuntimeError('%s Unable to load root folder (%s)' % (self.CLASS_TAG, root_folder))
        self.root_folder = root_folder
        self.collections = ProcessedImageCollection.load_many_from_folder(root_folder)
        if load_cwd_too:
            cwd_states = ProcessedImageCollection.restore_many_from_folder('.')
            # Update un-serialized classes taken from folder
            for restored_col in cwd_states:
                for col in self.collections:
                    if col.name == restored_col.name:
                        self.collections.remove(col)
                        break
                self.collections.append(restored_col)

    def get_by_name(self, name) -> Optional[ProcessedImageCollection]:
        for col in self.collections:
            if col.name == name:
                return col
        return None


# Global variables for easy image access
nodes = NodeCollections('/home/ljacobs/Argonne/car-data', load_cwd_too=True)


def write_metadata_for_img_folder(img_folder: str, name: str, restore_from_cwd=True, save_progress=True):
    """
    Takes a path to a folder of images (doesn't have to be a flat folder, can be a folder hierarchy too), processes
    the entirety of those images, and then writes a JSON metadata file in the root folder, making sure to save progress
    checkpoints along the way.

    The name parameter is used for saving checkpoints in progress.

    Steps
        - Pass through the image set once, recording each image's brightness value to determine whether this set
          has day and night peaks
        - Pass through some of images, training the background subtractor on them (if the image set has day images and
          night images, there will be two background subtractors used in order to greatly increase accuracy). The model
          only needs to train for about 150 images to get an adaquate estimate of the background.
        - Pass through the images a third time, building a list of images that are almost certainly background.
    """

    if restore_from_cwd:
        # Restore our processing progress in case of an error
        collection = ProcessedImageCollection.restore('%s.state' % name)
        if collection is None:
            print('Found no previous restore points')
            # Initialize and scan through the image folder
            collection = ProcessedImageCollection(img_folder, os.path.basename(img_folder))
        else:
            print('Restoring progress of: %s' % name)
    else:
        # Initialize and scan through the image folder
        collection = ProcessedImageCollection(img_folder, os.path.basename(img_folder))

    # Tag images in the collection with time of day data: either day or night
    collection.tag_images_with_tod(output=True)
    if save_progress:
        collection.save()

    # Tag images with foreground count
    print('[Tagging] Tagging images with foreground count values')

    if collection.local_images[0].get_tag('fg_count') is None and \
            collection.local_images[-1].get_tag('fg_count') is None:
        # Tag images if they have not been tagged before
        collection.tag_images_with_fg_count()
        if save_progress:
            collection.save()

    # Train the background subtractors
    collection.train_background_subtractors(n_for_each=150)
    if save_progress:
        collection.save()

    # This is the especially tricky part: choosing the threshold to consider images as being interesting.
    # For now, the script presents the user with a graph and tells him or her to manually enter in a threshold at their
    # judgment.

    collection.graph_foreground_counts()

    selected_threshold = None
    while True:
        try:
            selected_threshold = int(input('Select a threshold for filtering background images: '))
        except TypeError:
            continue
        break

    collection.tag_images_under_fg_threshold(selected_threshold)
    if save_progress:
        collection.save()
    collection.output_metadata_file(img_folder)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image_folder', type=str, help='The path to the input image folder / folder hierarchy')
    parser.add_argument('name', type=str, help='A short name for the image collection, ex. \"001e061182e8\"')
    parser.add_argument('-P', type=str, dest='save_option',
                        help='Save ProcessedImageCollection objects in state files for easy progress restoration '
                             'in case something goes wrong (Highly recommended)',
                        action='store_true')
    parser.add_argument()

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    write_metadata_for_img_folder(args.input_image_folder, args.name, save_progress=args.save_option)
