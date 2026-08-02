import { Component, inject, OnInit, signal } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { MatTableModule } from '@angular/material/table';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatCardModule } from '@angular/material/card';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatChipsModule } from '@angular/material/chips';
import { MatDialogModule, MatDialog } from '@angular/material/dialog';
import { DatePipe, NgClass, SlicePipe } from '@angular/common';
import { ReportsService } from '../../core/services/reports.service';
import { ValidationService } from '../../core/services/validation.service';
import { AuthService } from '../../core/services/auth.service';
import { NotificationService } from '../../core/services/notification.service';
import { ComplianceReport, ValidationJob } from '../../core/models';
import { GenerateReportDialogComponent } from './generate-report-dialog/generate-report-dialog';

@Component({
  selector: 'app-reports',
  imports: [
    ReactiveFormsModule, DatePipe, NgClass, SlicePipe,
    MatTableModule, MatButtonModule, MatIconModule,
    MatFormFieldModule, MatInputModule, MatSelectModule,
    MatCardModule, MatProgressSpinnerModule,
    MatTooltipModule, MatChipsModule, MatDialogModule,
  ],
  templateUrl: './reports.html',
  styleUrl: './reports.css',
})
export class ReportsComponent implements OnInit {
  private readonly svc       = inject(ReportsService);
  private readonly valSvc    = inject(ValidationService);
  private readonly route     = inject(ActivatedRoute);
  private readonly dialog    = inject(MatDialog);
  private readonly notify    = inject(NotificationService);
  readonly auth              = inject(AuthService);

  readonly loading  = signal(true);
  readonly signing  = signal<string | null>(null);  // report id being signed
  readonly deleting = signal<string | null>(null);  // report id being deleted
  readonly reports  = signal<ComplianceReport[]>([]);
  readonly jobs     = signal<ValidationJob[]>([]);

  readonly columns = ['created_at', 'job_id', 'report_type', 'status', 'signed', 'actions'];

  private _studyId?: string;

  ngOnInit(): void {
    this._studyId = this.route.snapshot.queryParamMap.get('study_id') ?? undefined;
    this.loadJobs();
    this.load();
  }

  load(): void {
    this.loading.set(true);
    this.svc.list({ study_id: this._studyId }).subscribe({
      next: r => { this.reports.set(r); this.loading.set(false); },
      error: () => this.loading.set(false),
    });
  }

  loadJobs(): void {
    this.valSvc.listJobs(this._studyId).subscribe({
      next: j => this.jobs.set(j.filter(x => x.status === 'completed')),
    });
  }

  openGenerate(): void {
    this.dialog
      .open(GenerateReportDialogComponent, { width: '460px', data: { jobs: this.jobs() } })
      .afterClosed()
      .subscribe(generated => { if (generated) this.load(); });
  }

  sign(report: ComplianceReport): void {
    if (!confirm('Apply your electronic signature to this report?')) return;
    this.signing.set(report.id);
    this.svc.sign(report.id).subscribe({
      next: updated => {
        this.reports.update(list => list.map(r => r.id === updated.id ? updated : r));
        this.signing.set(null);
        this.notify.success('Report signed successfully');
      },
      error: () => this.signing.set(null),
    });
  }

  delete(report: ComplianceReport): void {
    if (!confirm('Delete this report? This cannot be undone.')) return;
    this.deleting.set(report.id);
    this.svc.deleteReport(report.id).subscribe({
      next: () => {
        this.reports.update(list => list.filter(r => r.id !== report.id));
        this.deleting.set(null);
        this.notify.success('Report deleted');
      },
      error: () => this.deleting.set(null),
    });
  }

  download(report: ComplianceReport): void {
    this.svc.downloadBlob(report.id).subscribe(blob => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report_${report.id}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  signedLabel(r: ComplianceReport): string {
    if (!r.signed_at) return 'Unsigned';
    return new Date(r.signed_at).toLocaleDateString();
  }

  statusIcon(r: ComplianceReport): string {
    return r.storage_uri ? 'description' : 'hourglass_empty';
  }

  statusLabel(r: ComplianceReport): string {
    return r.storage_uri ? 'ready' : 'generating';
  }
}
