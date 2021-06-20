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

__title__ = 'scivae'
__description__ = ''
__url__ = 'https://github.com/ArianeMora/scivae.git'
__version__ = '1.0.0'
__author__ = 'Ariane Mora'
__author_email__ = 'ariane.n.mora@gmail.com'
__license__ = 'GPL3'

from scivae.loss import Loss
from scivae.vae import VAE
from scivae.vae import VAEException
from scivae.optimiser import Optimiser
from scivae.validate import Validate
from scivae.vis import Vis