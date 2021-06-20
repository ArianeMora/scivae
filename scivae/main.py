###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

import argparse
import os
import sys

from sciutil import SciUtil
from scivae import __version__, vae


def print_help():
    lines = ['-h Print help information.']
    print('\n'.join(lines))


def run(args):
    if args.t == 'd':
        c = Csv(args.l2g, chr_str=args.chr, start=args.start, end=args.end, value=args.value, header_extra=args.hdr,
                overlap_method=args.m, buffer_after_tss=args.downflank,
                buffer_before_tss=args.upflank, buffer_gene_overlap=args.overlap,
                gene_start=args.gstart, gene_end=args.gend, gene_chr=args.gchr,
                gene_direction=args.gdir, gene_name=args.gname
                )
        c.set_annotation_from_file(args.a)
        c.assign_locations_to_genes()  # Now we can run the assign values
        c.save_loc_to_csv(args.o)
        if args.b:
            c.convert_to_bed(c.loc_df, args.b, args.b)
    elif args.t == 'b':
        bed = Bed(args.l2g, overlap_method=args.m, buffer_after_tss=args.downflank,
                  buffer_before_tss=args.upflank, buffer_gene_overlap=args.overlap,
                  gene_start=args.gstart, gene_end=args.gend, gene_chr=args.gend,
                  gene_direction=args.gdir, gene_name=args.gname, chr_idx=args.chridx, start_idx=args.startidx,
                  end_idx=args.endidx, peak_value=args.valueidx, header_extra=args.hdridx
                  )
        # Add the gene annot
        bed.set_annotation_from_file(args.a)
        # Now we can run the assign values
        bed.assign_locations_to_genes()
        bed.save_loc_to_csv(args.o)


def gen_parser():
    parser = argparse.ArgumentParser(description='sciloc2gene')
    parser.add_argument('--configtype', type=str, default='s', help='Config type, either f (for json file) or s'
                                                                    ' (for json str).')
    parser.add_argument('--c', type=str, default='f', help='Config type, either f (for json file) or s'
                                                                    ' (for json str).')

    json_str = ""
    return parser


def main(args=None):
    parser = gen_parser()
    u = SciUtil()
    if args:
        sys.argv = args
    if len(sys.argv) == 1:
        print_help()
        sys.exit(0)
    elif sys.argv[1] in {'-v', '--v', '-version', '--version'}:
        print(f'scie2g v{__version__}')
        sys.exit(0)
    else:
        print(f'scie2g v{__version__}')
        args = parser.parse_args(args)
        # Validate the input arguments.
        if not os.path.isfile(args.a):
            u.err_p([f'The annotation file could not be located, file passed: {args.a}'])
            sys.exit(1)
        if not os.path.isfile(args.l2g):
            u.err_p([f'The input file could not be located, file passed: {args.l2g}'])
            sys.exit(1)
        if args.t != 'b' and args.t != 'd':
            u.err_p([f'The file type passed is not supported: {args.t}, '
                     f'filetype must be "b" for bed or "d" for dmrseq.'])
            sys.exit(1)
        # Otherwise we have need successful so we can run the program
        u.dp(['Running scie2g on input file: ', args.l2g,
              '\nWith annotation file: ', args.a,
              '\nSaving to output file: ', args.o,
              '\nOverlap method:', args.m,
              '\nUpstream flank: ', args.upflank,
              '\nDownstream flank:', args.downflank,
              '\nGene overlap: ', args.overlap])
        u.warn_p(['Assuming your annotation file and your input file are SORTED!'])
        # RUN!
        run(args)
    # Done - no errors.
    sys.exit(0)


if __name__ == "__main__":
    main()
    # ----------- Example below -----------------------
    """
    root_dir = '../sciloc2gene/'
    main(["--a", f'{root_dir}data/hsapiens_gene_ensembl-GRCh38.p13.csv',
          "--o", f'{root_dir}output_file.csv',
          "--l2g", f'{root_dir}tests/data/test_H3K27ac.bed',
          "--t", "b",
          "--upflank", "3000"])

    root_dir = '../'
    main(["--a", f'{root_dir}data/hsapiens_gene_ensembl-GRCh38.p13.csv', # mmusculus_gene_ensembl-GRCm38.p6.csv', #
      "--o", f'{root_dir}output_file_2.csv',
      "--l2g", f'{root_dir}tests/data/test_dmrseq.csv',
      "--t", "d",
      "--upflank", "20", "--chr", "seqnames", "--value", "stat"])
    """
