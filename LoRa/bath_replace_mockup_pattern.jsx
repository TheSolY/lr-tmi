var srcFolder = new Folder('/Users/solyarkoni-port-mac/PycharmProjects/textual_inversion/Coffee Mug for LoRa/patterns');
var outputFolder = new Folder('/Users/solyarkoni-port-mac/PycharmProjects/textual_inversion/Coffee_Mug_Dataset_printful');

var searchMask = '*.???'
var fileList = srcFolder.getFiles(searchMask);

var doc = activeDocument;
var curLayer = doc.activeLayer;
var soName
var jpgOptions = new JPEGSaveOptions();
jpgOptions.quality = 8;

for (var i=0;i<fileList.length;i++){
    replaceSO (fileList[i]);
    var fName = fileList[i].name.split('.')[0];
    dupeFile ();
    var doc2 = activeDocument;
    // doc2.saveAs(new File(outputFolder +'/'+fName + '1.png'));
    var saveFile = File(outputFolder +'/'+fName + '1.png');
    sfwPNG24(saveFile);
    doc2.close(SaveOptions.DONOTSAVECHANGES);
    }

function replaceSO(file){
    var idplacedLayerReplaceContents = stringIDToTypeID( "placedLayerReplaceContents" );
        var desc5 = new ActionDescriptor();
        var idnull = charIDToTypeID( "null" );
        desc5.putPath( idnull, new File( file ) );
    executeAction( idplacedLayerReplaceContents, desc5, DialogModes.NO );
    }

function dupeFile(){
    var idDplc = charIDToTypeID( "Dplc" );
        var desc11 = new ActionDescriptor();
        var idnull = charIDToTypeID( "null" );
            var ref1 = new ActionReference();
            var idDcmn = charIDToTypeID( "Dcmn" );
            var idOrdn = charIDToTypeID( "Ordn" );
            var idFrst = charIDToTypeID( "Frst" );
            ref1.putEnumerated( idDcmn, idOrdn, idFrst );
        desc11.putReference( idnull, ref1 );
        var idMrgd = charIDToTypeID( "Mrgd" );
        desc11.putBoolean( idMrgd, true );
    executeAction( idDplc, desc11, DialogModes.NO );
    }



function sfwPNG24(saveFile){
var pngOpts = new PNGSaveOptions;
pngOpts.compression = 9;
pngOpts.interlaced = false;
activeDocument.saveAs(saveFile, pngOpts, true, Extension.LOWERCASE);
}
