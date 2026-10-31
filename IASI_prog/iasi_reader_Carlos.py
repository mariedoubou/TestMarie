import pathlib
import re
from enum import IntEnum, IntFlag

import numpy as np

# constants
EPOCH = np.datetime64('2000-01-01T00:00:00.000')
# IASI specific
AMCO = 100
AMLI = 100
CCD = 2
IMCO = 64
IMLI = 64
MAXBA = 3600
N = 221000
NBK = 6
NCL = 7
NIM = 28
PN = 4
SB = 3
SGI = 25
SNOT = 30
SNOTp4 = 34
SS = 8700
VP = 1

# General types
class RecordClass(IntEnum):
    RESERVED = 0
    MPHR = 1  # Main Product Header Record
    SPHR = 2  # Secondary Product Header Record
    IPR = 3  # Internal Pointer Record
    GEADR = 4  # Global External Auxiliary Data Record
    GIADR = 5  # Global Internal Auxiliary Data Record
    VEADR = 6  # Variable External Auxiliary Data Record
    VIADR = 7  # Variable Internal Auxiliary Data Record
    MDR = 8  # Measurement Data Record

class InstrumentGroup(IntEnum):
    GENERIC = 0
    AMSU_A = 1
    ASCAT = 2
    ATOVS = 3
    AVHRR3 = 4
    GOME = 5
    GRAS = 6
    HIRS4 = 7
    IASI = 8
    MHS = 9
    SEM = 10
    ADCS = 11
    SBUV = 12
    DUMMY = 13
    ARCHIVE = 99
    IASI_L2 = 15

t_time_cds_short = np.dtype([
    ('Day', '>u2'),
    ('MillisecondsOfDay', '>u4')
])
t_time_cds_long = np.dtype([
    ('Day', '>u2'),
    ('MillisecondsOfDay', '>u4'),
    ('MicrosecondsOfMilliseconds', '>u2')
])
t_time_cds_expanded = np.dtype([
    ('Day', '>u2'),
    ('MillisecondsOfDay', '>u4'),
    ('MicrosecondsOfMilliseconds', '>u2'),
    ('NanosecondsOfMicroseconds', '>u2')
])
t_time_generalized = 'S15'
t_generic_record_header = np.dtype([
    ("RecordClass", "u1"),
    ("InstrumentGroup", "u1"),
    ("RecordSubclass", "u1"),
    ("RecordSubclassVersion", "u1"),
    ("RecordSize", ">u4"),
    ("RecordStartTime", t_time_cds_short),
    ("RecordStopTime", t_time_cds_short)
])
t_generic_record_pointer = np.dtype([
    ("TargetRecordClass", "u1"),
    ("TargetInstrumentGroup", "u1"),
    ("TargetRecordSubclass", "u1"),
    ("TargetRecordOffset", ">u4"),
])
t_obt2utc = np.dtype([
    ('Utc0', '>u2', 4),
    ('CcuObt0', '>u2', 3),
    ('ClockStep', '>u2', 2)
])
t_vint4 = np.dtype([("scale", "i1"), ("scaled", ">i4")])

# GIADR
class RecordSubClassGIADR(IntEnum):
    QUALITY = 0
    SCALE_FACTORS = 1

t_giadr_quality = np.dtype([
    ("IDefPsfSondNbLin", ">i4", PN),
    ("IDefPsfSondNbCol", ">i4", PN),
    ("IDefPsfSondOverSampFactor", t_vint4),
    ("IDefPsfSondY", ">i4", (PN, 100)),
    ("IDefPsfSondZ", ">i4", (PN, 100)),
    ("IDefPsfSondWgt", t_vint4, (PN, 100, 100)),
    ("IDefllSSrfNsfirst", ">i4"),
    ("IDefllSSrfNslast", ">i4"),
    ("IDefllSSrf", t_vint4, 100),
    ("IDefllSSrfDWn", t_vint4),
    ("IDefIISNeDT", t_vint4, (IMLI, IMCO)),
    ("IDefDptIISDeadPix", "?", (IMLI, IMCO))
])

t_giadr_scale_factors = np.dtype([
    ("IDefScaleSondNbScale", ">i2"),
    ("IDefScaleSondNsfirst", ">i2", 10),
    ("IDefScaleSondNslast", ">i2", 10),
    ("IDefScaleSondScaleFactor", ">i2", 10),
    ("IDefScaleIISScaleFactor", ">i2"),
])


# MDR
class RecordSubClassMDR(IntEnum):
    MDR1A = 0
    MDR1B = 1
    MDR1C = 2

class QualityFlagDetailed(IntFlag):
    Hardware = 1 << 0
    Band1Spikes = 1 << 1
    Band2Spikes = 1 << 2
    Band3Spikes = 1 << 3
    NZPDComplexCalibrationError = 1 << 4
    OnBoardGeneralQualityFlag = 1 << 5
    OverflowUnderflow = 1 << 6
    SpectralCalibrationError = 1 << 7
    RadiometricPostCalibrationError = 1 << 8
    GQisFlagQualSummaryFlagAllBands = 1 << 9
    MissingSounderData = 1 << 10
    MissingIISData = 1 << 11
    MissingAVHRRData = 1 << 12

t_generic_quality_indicators = np.dtype([
    ("DEGRADED_INST_MDR", "?"),
    ("DEGRADED_PROC_MDR", "?"),
])

t_level1data = np.dtype([
    ("GEPSIasiMode", "V4"),
    ("GEPSOPSProcessingMode", "V4"),
    ("GEPSIdConf", "V32"),
    ("GEPSLocIasiAvhrr_IASI", t_vint4, (SNOT, PN, 2)),
    ("GEPSLocIasiAvhrr_IIS", t_vint4, (SNOT, SGI, 2)),
    ("OBT", "V6", SNOT),
    ("OnboardUTC", t_time_cds_short, SNOT),
    ("GEPSDatIasi", t_time_cds_short, SNOT),
    ("GIsfLinOrigin", ">i4", CCD),
    ("GIsfColOrigin", ">i4", CCD),
    ("GIsfPds1", ">i4", CCD),
    ("GIsfPds2", ">i4", CCD),
    ("GIsfPds3", ">i4", CCD),
    ("GIsfPds4", ">i4", CCD),
    ("GEPS_CCD", "?", SNOT),
    ("GEPS_SP", ">i4", SNOT),
    ("GIrcImage", ">u2", (SNOT, IMLI, IMCO)),
    ("GQisFlagQual", "?", (SNOT, PN, SB)),
    ("GQisFlagQualDetailed", ">u2", (SNOT, PN)),
    ("GQisQualIndex", t_vint4),
    ("GQisQualIndexIIS", t_vint4),
    ("GQisQualIndexLoc", t_vint4),
    ("GQisQualIndexRad", t_vint4),
    ("GQisQualIndexSpect", t_vint4),
    ("GQisSysTecIISQual", ">u4"),
    ("GQisSysTecSondQual", ">u4"),
    ("GGeoSondLoc", ">i4", (SNOT, PN, 2)),
    ("GGeoSondAnglesMETOP", ">i4", (SNOT, PN, 2)),
    ("GGeoIISAnglesMETOP", ">i4", (SNOT, SGI, 2)),
    ("GGeoSondAnglesSUN", ">i4", (SNOT, PN, 2)),
    ("GGeoIISAnglesSUN", ">i4", (SNOT, SGI, 2)),
    ("GGeoIISLoc", ">i4", (SNOT, SGI, 2)),
    ("EARTH_SATELLITE_DISTANCE", ">u4"),
    # level 1c specific
    ("IDefSpectDWn1b", t_vint4),
    ("IDefNsfirst1b", ">i4"),
    ("IDefNslast1b", ">i4"),
    ("GS1cSpect", ">i2", (SNOT, PN, SS)),
    ("IDefCovarMatEigenVal1c", t_vint4, (100, CCD)),
    ("IDefCcsChannelId", ">i4", NBK),
    ("GCcsRadAnalNbClass", ">i4", (SNOT, PN)),
    ("GCcsRadAnalWgt", t_vint4, (SNOT, PN, NCL)),
    ("GCcsRadAnalY", ">i4", (SNOT, PN, NCL)),
    ("GCcsRadAnalZ", ">i4", (SNOT, PN, NCL)),
    ("GCcsRadAnalMean", t_vint4, (SNOT, PN, NCL, NBK)),
    ("GCcsRadAnalStd", t_vint4, (SNOT, PN, NCL, NBK)),
    ("GCcsImageClassified", "u1", (SNOT, AMLI, AMCO)),
    ("IDefCcsMode", "V4"),
    ("GCcsImageClassifiedNbLin", ">i2", SNOT),
    ("GCcsImageClassifiedNbCol", ">i2", SNOT),
    ("GCcsImageClassifiedFirstLin", t_vint4, SNOT),
    ("GCcsImageClassifiedFirstCol", t_vint4, SNOT),
    ("GCcsRadAnalType", "?", (SNOT, NCL)),
    ("GIacVarImagIIS", t_vint4, SNOT),
    ("GIacAvgImagIIS", t_vint4, SNOT),
    ("GEUMAvhrr1BCldFrac", "u1", (SNOT, PN)),
    ("GEUMAvhrr1BLandFrac", "u1", (SNOT, PN)),
    ("GEUMAvhrr1BQual", "u1", (SNOT, PN))
])

t_mdr_1c = np.dtype([
    ("RecordHeader", t_generic_record_header),
    ("QualityIndicators", t_generic_quality_indicators),
    ("Level1Data", t_level1data),
])

angle_scale_factor = np.float32(1e-6)

_mphr_extractor = re.compile(r"(\w+)\s*=\s*(-?\d*\w*)", flags=re.MULTILINE)

class NativeIASI1c:
    def __init__(self, filepath):
        self.filepath = pathlib.Path(filepath)
        self._scan_file()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.filepath})"

    def _scan_file(self):
        with self.filepath.open(mode="rb") as file:
            while True:
                grh_buffer = file.read(t_generic_record_header.itemsize)
                grh, = np.frombuffer(grh_buffer, dtype=t_generic_record_header, count=1)
                record_class = grh['RecordClass']
                record_size = grh['RecordSize']
                size = record_size - t_generic_record_header.itemsize
                buffer = file.read(size)
                if record_class == RecordClass.MPHR:
                    self._mphr_bytes = buffer
                elif record_class == RecordClass.IPR:
                    ipr, = np.frombuffer(buffer, dtype=t_generic_record_pointer)
                    assert ipr["TargetInstrumentGroup"] == InstrumentGroup.IASI
                    offset = ipr["TargetRecordOffset"]
                    if ipr["TargetRecordClass"] == RecordClass.GIADR:
                        offset += t_generic_record_header.itemsize
                        if ipr["TargetRecordSubclass"] == RecordSubClassGIADR.QUALITY:
                            self._giadr_quality = np.memmap(
                                self.filepath, dtype=t_giadr_quality, mode="r", offset=offset, shape=1
                            )
                        elif ipr["TargetRecordSubclass"] == RecordSubClassGIADR.SCALE_FACTORS:
                            self._giadr_scale_factors = np.memmap(
                                self.filepath, dtype=t_giadr_scale_factors, mode="r", offset=offset, shape=1
                            )
                        else:
                            raise RuntimeError("Unkonwn GIADR Subclass!")
                    elif ipr["TargetRecordClass"] == RecordClass.MDR:
                        assert ipr["TargetRecordSubclass"] == RecordSubClassMDR.MDR1C
                        self._mdrs = np.memmap(self.filepath, dtype=t_mdr_1c, mode="r", offset=offset)
                else:
                    break

    @property
    def mphr(self):
        return dict(_mphr_extractor.findall(self._mphr_bytes.decode("ascii")))

    @property
    def scan_time(self):
        day = self._mdrs["Level1Data"]["GEPSDatIasi"]["Day"]
        ms = self._mdrs["Level1Data"]["GEPSDatIasi"]["MillisecondsOfDay"]
        step_times = EPOCH + day * np.timedelta64(1, "D") + ms * np.timedelta64(1, "ms")
        # make the shape similar to latitude and longitude
        return np.lib.stride_tricks.as_strided(
            step_times,
            shape=step_times.shape+(PN,),
            strides=step_times.strides+(0,),
            writeable=False
        )

    @property
    def longitude(self):
        return self._mdrs["Level1Data"]["GGeoSondLoc"][:, :, :, 0].astype(np.float32) * angle_scale_factor

    @property
    def latitude(self):
        return self._mdrs["Level1Data"]["GGeoSondLoc"][:, :, :, 1].astype(np.float32) * angle_scale_factor

    @property
    def viewing_zenith_angle(self):
        return self._mdrs["Level1Data"]["GGeoSondAnglesMETOP"][:, :, :, 0].astype(np.float32) * angle_scale_factor

    @property
    def viewing_azimuth_angle(self):
        return self._mdrs["Level1Data"]["GGeoSondAnglesMETOP"][:, :, :, 1].astype(np.float32) * angle_scale_factor

    @property
    def solar_zenith_angle(self):
        return self._mdrs["Level1Data"]["GGeoSondAnglesSUN"][:, :, :, 0].astype(np.float32) * angle_scale_factor

    @property
    def solar_azimuth_angle(self):
        return self._mdrs["Level1Data"]["GGeoSondAnglesSUN"][:, :, :, 1].astype(np.float32) * angle_scale_factor

    @property
    def cloud_fraction(self):
        return self._mdrs["Level1Data"]["GEUMAvhrr1BCldFrac"]

    @property
    def land_fraction(self):
        return self._mdrs["Level1Data"]["GEUMAvhrr1BLandFrac"]

    @property
    def pixel_quality_flag(self):
        return self._mdrs["Level1Data"]["GQisFlagQualDetailed"]

    @property
    def wavenumbers(self):
        IDefSpectDWn1b = (
            self._mdrs["Level1Data"]["IDefSpectDWn1b"]["scaled"] 
            / 10.0**self._mdrs["Level1Data"]["IDefSpectDWn1b"]["scale"]
        )
        IDefNsfirst1b = self._mdrs["Level1Data"]["IDefNsfirst1b"]
        IDefNslast1b = self._mdrs["Level1Data"]["IDefNslast1b"]
        sample_width = IDefSpectDWn1b[0]
        first = IDefNsfirst1b[0]
        last = IDefNslast1b[0]
        assert (IDefSpectDWn1b == sample_width).all()
        assert (IDefNsfirst1b == first).all()
        assert (IDefNslast1b == last).all()
        # minus one as in Product Format Specification
        # there we start at k=1 and subtract 2
        return (sample_width * np.arange(first - 1, last) / 100).astype(np.float32)

    @property
    def spectra(self):
        return self.get_spectra()

    def get_spectra(self, lines=slice(None)):
        # spectra of the selected scan lines only (the whole orbit is ~3 GB in float32)
        giadr_scale_factors = self._giadr_scale_factors[0]
        nsplit = giadr_scale_factors["IDefScaleSondNbScale"]
        first = self._mdrs["Level1Data"]["IDefNsfirst1b"][0]
        starts = giadr_scale_factors["IDefScaleSondNsfirst"][:nsplit] - first
        stops = giadr_scale_factors["IDefScaleSondNslast"][:nsplit] - first + 1
        scale = np.float32(10)**np.repeat(
            -giadr_scale_factors["IDefScaleSondScaleFactor"][:nsplit], 
            stops - starts
        )
        return scale * self._mdrs["Level1Data"]["GS1cSpect"][lines, ..., starts[0]:stops[-1]]

    def to_dataset(self, angles=True, qflags=True, frac=True):
        import xarray as xr
        # Note: I could use an in memory-Zarr to compress the data
        dimensions = ["line", "fov", "pixel"]
        ds = xr.Dataset(
            data_vars=dict(
                longitude=(dimensions, self.longitude),
                latitude=(dimensions, self.latitude),
                pixel_time=(dimensions, self.scan_time),
                spectrum=(dimensions + ["wavenumber"], self.spectra)
            ),
            coords=dict(
                time=(["line"], self.scan_time[:, 0, 0]),
                wavenumber=self.wavenumbers,
            )
        )
        ds["time"].attrs["standard_name"] = "time"
        ds["wavenumber"].attrs["units"] = "m-1"
        ds["longitude"].attrs["standard_name"] = "longitude"
        ds["longitude"].attrs["units"] = "degrees_east"
        ds["latitude"].attrs["standard_name"] = "latitude"
        ds["latitude"].attrs["units"] = "degrees_north"
        ds["pixel_time"].attrs["long_name"] = "pixel acquisition time"
        ds['spectrum'].attrs["long_name"] = "IASI spectrum"
        ds['spectrum'].attrs["units"] = "W/m2/sr/m-1"
        if angles:
            ds["viewing_zenith_angle"] = (dimensions, self.viewing_zenith_angle)
            ds["viewing_zenith_angle"].attrs["long_name"] = "satellite_zenith_angle"
            ds["viewing_zenith_angle"].attrs["standard_name"] = "sensor_zenith_angle"
            ds["viewing_zenith_angle"].attrs["units"] = "degrees"
            ds["viewing_azimuth_angle"] = (dimensions, self.viewing_azimuth_angle)
            ds["viewing_azimuth_angle"].attrs["long_name"] = "satellite_azimuth_angle"
            ds["viewing_azimuth_angle"].attrs["standard_name"] = "sensor_azimuth_angle"
            ds["viewing_azimuth_angle"].attrs["units"] = "degrees"
            ds["solar_zenith_angle"] = (dimensions, self.solar_zenith_angle)
            ds["solar_zenith_angle"].attrs["long_name"] = "solar_zenith_angle"
            ds["solar_zenith_angle"].attrs["standard_name"] = "solar_zenith_angle"
            ds["solar_zenith_angle"].attrs["units"] = "degrees"
            ds["solar_azimuth_angle"] = (dimensions, self.solar_azimuth_angle)
            ds["solar_azimuth_angle"].attrs["long_name"] = "solar_azimuth_angle"
            ds["solar_azimuth_angle"].attrs["standard_name"] = "solar_azimuth_angle"
            ds["solar_azimuth_angle"].attrs["units"] = "degrees"
        if qflags:
            ds["pixel_quality_flag"] = (dimensions, self.pixel_quality_flag)
            ds["pixel_quality_flag"].attrs["long_name"] = "pixel_quality_flag"
            ds["pixel_quality_flag"].attrs["flag_masks"] = np.array(QualityFlagDetailed, dtype=self.pixel_quality_flag.dtype)
            ds["pixel_quality_flag"].attrs["flag_meanings"] = " ".join(qflag.name for qflag in QualityFlagDetailed)
        if frac:
            ds["cloud_fraction"] = (dimensions, self.cloud_fraction)
            ds["cloud_fraction"].attrs["long_name"] = "avhrr_cloud_fraction"
            ds["cloud_fraction"].attrs["units"] = "%"
            ds["land_fraction"] = (dimensions, self.land_fraction)
            ds["land_fraction"].attrs["long_name"] = "avhrr_land_fraction"
            ds["land_fraction"].attrs["units"] = "%"
        return ds
