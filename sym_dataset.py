class VOC:
    def __init__(self, is_train):
        from symimdb.pascal_voc import PascalVOC
        self._is_train = is_train
        self._ds_cls = PascalVOC
        self.default_imageset = '2007_trainval' if is_train else '2007_test'

    def set_args(self, args):
        args.rcnn_num_classes = len(self._ds_cls.classes)

    def get_dataset(self, imageset):
        imdb = self._ds_cls(imageset, 'data', 'data/VOCdevkit')
        if self._is_train:
            imdb.filter_roidb()
            imdb.append_flipped_images()
        return imdb

    def get_names(self):
        return self._ds_cls.classes


class COCO:
    def __init__(self, is_train):
        from symimdb.coco import coco
        self._is_train = is_train
        self._ds_cls = coco
        self.default_imageset = 'train2017' if is_train else 'val2017'

    def set_args(self, args):
        args.rpn_anchor_scales = (2, 4, 8, 16, 32)
        args.rcnn_num_classes = len(self._ds_cls.classes)

    def get_dataset(self, imageset):
        imdb = self._ds_cls(imageset, 'data', 'data/coco')
        if self._is_train:
            imdb.filter_roidb()
            imdb.append_flipped_images()
        return imdb

    def get_names(self):
        return self._ds_cls.classes


class DatasetFactory:
    DATASETS = {
        'voc': VOC,
        'coco': COCO
    }
    def __init__(self, name):
        if name not in self.DATASETS:
            raise ValueError("dataset {} not supported".format(name))
        self._ds_cls = self.DATASETS[name]

    def get_train(self, args):
        ds = self._ds_cls(is_train=True)
        ds.set_args(args)
        imageset = args.imageset if args.imageset else ds.default_imageset
        isets = imageset.split('+')
        roidb = []
        for iset in isets:
            imdb = ds.get_dataset(iset)
            roidb.extend(imdb.roidb)
        return roidb

    def get_test(self, args):
        ds = self._ds_cls(is_train=False)
        ds.set_args(args)
        imageset = args.imageset if args.imageset else ds.default_imageset
        imdb = ds.get_dataset(imageset)
        return imdb

    def get_demo(self, args):
        ds = self._ds_cls(is_train=False)
        ds.set_args(args)
        names = ds.get_names()
        return names